# Build IMath and run tests with GCC on Linux.
#
# Usage (from the imath root):
#
#   docker run --rm -it "$(docker build -f tests/linux/Dockerfile -q .)"
#
FROM docker.io/alpine:latest AS base

RUN apk add --no-cache bash build-base gcc gmp-dev make python2

FROM base AS test
COPY . /imath
WORKDIR /imath
CMD make distclean examples check
