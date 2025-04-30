#!/bin/bash -e

# Build and install zlib using provided nextsilicon sysroot

if [ $# -ne 1 ] ; then
	echo "Usage: $0 <NEXT_HOME>" >&2
	exit 1
fi

NEXT_HOME=$1

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC=${NPROC_CI:-$(nproc)}
SYSROOT=${NEXT_HOME}/sysroot/usr
export PATH=${NEXT_HOME}/bin:$PATH

export CC=nextcc
export CXX=nextcxx
export AR=${NEXT_HOME}/llvm/bin/ar
export RANLIB=${NEXT_HOME}/llvm/bin/ranlib
export LDFLAGS="--no-next-binfmt-magic-number"

mkdir -p zlib
cd zlib

ZLIB_VER=1.2.13
ZLIB_DOWNLOAD_URL="https://github.com/madler/zlib/releases/download/v${ZLIB_VER}/zlib-${ZLIB_VER}.tar.xz"
STAMP_FILE=$(echo -n $ZLIB_DOWNLOAD_URL | md5sum | cut -d" " -f1).download_stamp
if [ ! -f $STAMP_FILE ] ; then
	curl -L $ZLIB_DOWNLOAD_URL | tar -xJf -
	touch $STAMP_FILE
fi

cd "zlib-${ZLIB_VER}"

./configure --prefix=''
make DESTDIR=${SYSROOT} -j $NPROC install
