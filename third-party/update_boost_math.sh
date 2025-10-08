#!/usr/bin/env bash

set -e # stop at the first error

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

echo "This script deletes ${SCRIPT_DIR}/boost-math and re-downloads it from the standalone Boost.Math release."
echo "It then subsets it so it only contains the parts that are used in libc++."
echo
read -p "To continue, please enter the version you wish to download (e.g. 1.89.0), or Ctrl+C to cancel: " version

# Boost versions are always 1.xx.yy so far, use this to validate.
# That will probably have to be updated at some point.
if [[ "${version}" != "1."* ]]; then
    echo "Invalid version '${version}' provided, was expecting 1.XX.YY"
    exit 1
fi

echo "****************************************"
echo "Downloading Boost.Math ${version}"
echo "****************************************"
BOOST_URL="https://github.com/boostorg/math/archive/refs/tags/boost-${version}.tar.gz"
function cleanup_tarball() {
    rm ${SCRIPT_DIR}/boost-math-${version}.tar.gz
}
trap cleanup_tarball EXIT
wget "${BOOST_URL}" -O ${SCRIPT_DIR}/boost-math-${version}.tar.gz
rm -rf ${SCRIPT_DIR}/boost-math
mkdir ${SCRIPT_DIR}/boost-math
tar -x --file ${SCRIPT_DIR}/boost-math-${version}.tar.gz -C ${SCRIPT_DIR}/boost-math --strip-components=1

echo "****************************************"
echo "Subsetting Boost.Math ${version}"
echo "****************************************"
rm -rf ${SCRIPT_DIR}/boost-math/{.circleci,.drone,.github,build,config,doc,example,meta,test,tools}
