# Build tool dependency policy

To ensure the broadest compatibility when building the benchmark library, but
still allow forward progress, we require any build tooling to be available for:

* Debian stable _and_
* The last two Ubuntu LTS releases

Currently, this means using build tool versions that are available for Ubuntu
18.04 (Bionic Beaver), Ubuntu 20.04 (Focal Fossa), and Debian 11 (bullseye).

_Note, CI also runs ubuntu-16.04 and ubuntu-14.04 to ensure best effort support
for older versions._

## cmake
The current supported version is cmake 3.5.1 as of 2018-06-06.

_Note, this version is also available for Ubuntu 14.04, an older Ubuntu LTS
release, as `cmake3`._
