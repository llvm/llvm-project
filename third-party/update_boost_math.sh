#!/usr/bin/env bash

set -e # stop at the first error

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
VERSION="1.89.0"

echo "This script deletes ${SCRIPT_DIR}/boost-math and re-downloads it from the standalone Boost.Math release version ${VERSION}."
echo "It then subsets it so it only contains the parts that are used in libc++."
echo
read -p "Press a key to continue, or Ctrl+C to cancel"

echo "****************************************"
echo "Downloading Boost.Math ${VERSION}"
echo "****************************************"
BOOST_URL="https://github.com/boostorg/math/archive/refs/tags/boost-${VERSION}.tar.gz"
function cleanup_tarball() {
    rm ${SCRIPT_DIR}/boost-math-${VERSION}.tar.gz
}
trap cleanup_tarball EXIT
wget "${BOOST_URL}" -O ${SCRIPT_DIR}/boost-math-${VERSION}.tar.gz
rm -rf ${SCRIPT_DIR}/boost-math
mkdir ${SCRIPT_DIR}/boost-math
tar -x --file ${SCRIPT_DIR}/boost-math-${VERSION}.tar.gz -C ${SCRIPT_DIR}/boost-math --strip-components=1

echo "****************************************"
echo "Subsetting Boost.Math ${VERSION}"
echo "****************************************"
rm -rf ${SCRIPT_DIR}/boost-math/{.circleci,.drone,.github,build,config,doc,example,meta,reporting,src,test,tools}

echo "****************************************"
echo "Patching Boost.Math ${VERSION} for libc++"
echo "****************************************"
# Mark boost_math include dir as SYSTEM so libc++ consumers don't surface warnings from vendored boost-math code (e.g.
# -Wdeprecated-redundant-constexpr-static-def). Upstream keeps these for pre-C++17 compat.
sed -i 's|target_include_directories(boost_math INTERFACE include)|target_include_directories(boost_math SYSTEM INTERFACE include)|' \
    ${SCRIPT_DIR}/boost-math/CMakeLists.txt

# Verify the patch landed -- fail loudly if upstream renamed the target.
grep -q 'target_include_directories(boost_math SYSTEM INTERFACE include)' \
    ${SCRIPT_DIR}/boost-math/CMakeLists.txt \
    || { echo "ERROR: SYSTEM include patch failed -- upstream CMakeLists.txt structure changed"; exit 1; }
