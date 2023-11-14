# Packaging

This directory contains utilities to package the project along with relevant
dependencies and toolchains to build the dataset.

### Building the Docker Image

To build the Docker image, run the following command from the root of the
repository:

```bash
docker build -t llvm-ir-dataset-utils -f ./.packaging/Dockerfile .
```

To get the image building on machines where an older firewall or custom SSL
certificates are used, you can pass the following two build arguments to
the Docker build to make the image build work in your environment:

* `CUSTOM_CERT` - Pass the path to a `*.crt` file in the build context to make
the container use the certificate. Note that the file extension must be `*.crt`
and not `*.pem` or something else due to how Ubuntu's `update-ca-certificates`
detects new certificates.
* `ENABLE_LEGACY_RENEGOTIATION` - Enables legacy renegotiation which is a
problem on some systems that have a firewall in place when accessing certain
hosts.

As an example, to build a container in an environment that doesn't support SSL
renegotiation and with a custom certificate, you can run the following commands:

1. Start by making sure your current working directory is the root of the
project:
```bash
cd /path/to/llvm-ir-dataset-utils
```
2. Copy over the certificate (bundle) that you want the container to use:
```bash
cp /path/to/certificate.crt ./additional_cert.crt
```
3. Build the container image, making sure to specify the appropriate build
flags:
```bash
docker build \
  -t llvm-ir-dataset-utils \
  -f ./.packaging/Dockerfile \
  --build-arg="CUSTOM_CERT=./additional_cert.crt" \
  --build-arg="ENABLE_LEGACY_RENEGOTIATION=ON"
```

Then you should end up with the desired container image.
