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
# Guard Boost.Math's iostream-only exception-message formatting (prec_format / <sstream>)
# under BOOST_MATH_NO_EXCEPTIONS, so the no-exceptions back-end builds without <sstream>.
# Needed for libc++ configurations without localization. Drop once fixed upstream.
patch -p1 -d ${SCRIPT_DIR} < ${SCRIPT_DIR}/boost-math-libcxx.patch

# Verify the patch landed -- fail loudly if upstream restructured error_handling.hpp.
grep -q 'prec_format is only used to build exception' \
    ${SCRIPT_DIR}/boost-math/include/boost/math/policies/error_handling.hpp \
    || { echo "ERROR: no-exceptions iostream patch failed -- upstream error_handling.hpp structure changed"; exit 1; }

# The snapshot's own CMakeLists.txt is not used by the LLVM build: cmake/Modules/FindBoostMath.cmake
# declares the `boost_math` INTERFACE target instead, so the include dir can be marked SYSTEM without
# patching upstream. Fail loudly if upstream's interface grows beyond what that module mirrors.
grep -q 'target_compile_definitions(boost_math INTERFACE BOOST_MATH_STANDALONE=1)' \
    ${SCRIPT_DIR}/boost-math/CMakeLists.txt \
    || { echo "ERROR: upstream boost_math interface changed -- re-check cmake/Modules/FindBoostMath.cmake"; exit 1; }
