# System construction



## Use dockerfile to build systems
1.Use the dockerfile example we provided after modification or customize Dockerfile by yourself.

2.Using the command line: run the following command:

```
docker build -t {image_name} .
docker run -it --name {container_name} -p {Container_port}:{native port} -d {image_name}
```

3.Use port mapping to access docker or execute the following commands to access through bash:

```
docker exec -it {container_name} /bin/bash
```

**Note**: The dependency and deployment method of different versions of the software system may change. Please confirm the relevant information of the current version before deployment

