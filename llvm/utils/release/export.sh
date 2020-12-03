#!/bin/sh
#===-- tag.sh - Tag the LLVM release candidates ----------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# Create branches and release candidates for the LLVM release.
#
#===------------------------------------------------------------------------===#

set -e

projects="llvm clang compiler-rt libcxx libcxxabi libclc clang-tools-extra polly lldb lld openmp libunwind flang"

release=""
rc=""

usage() {
    echo "Export the Git sources and build tarballs from them"
    echo "usage: `basename $0`"
    echo " "
    echo "  -release <num> The version number of the release"
    echo "  -rc <num>      The release candidate number"
    echo "  -final         The final tag"
}

export_sources() {
    release_no_dot=`echo $release | sed -e 's,\.,,g'`
    tag="llvmorg-$release"

    if [ "$rc" = "final" ]; then
        rc=""
    else
        tag="$tag-$rc"
    fi

    llvm_src_dir=$(readlink -f $(dirname "$(readlink -f "$0")")/../../..)
    [ -d $llvm_src_dir/.git ] || ( echo "No git repository at $llvm_src_dir" ; exit 1 )

    echo $tag
    target_dir=$(pwd)

    echo "Creating tarball for llvm-project ..."
    pushd $llvm_src_dir/
    git archive --prefix=llvm-project-$release$rc.src/ $tag . | xz >$target_dir/llvm-project-$release$rc.src.tar.xz
    popd

    if [ ! -d test-suite-$release$rc.src ]
    then
      echo "Fetching LLVM test-suite source ..."
      mkdir -p test-suite-$release$rc.src
      curl -L https://github.com/llvm/test-suite/archive/$tag.tar.gz | \
          tar -C test-suite-$release$rc.src --strip-components=1 -xzf -
    fi
    echo "Creating tarball for test-suite ..."
    tar --sort=name --owner=0 --group=0 \
        --pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime \
        -cJf test-suite-$release$rc.src.tar.xz test-suite-$release$rc.src

    for proj in $projects; do
        echo "Creating tarball for $proj ..."
        pushd $llvm_src_dir/$proj
        git archive --prefix=$proj-$release$rc.src/ $tag . | xz >$target_dir/$proj-$release$rc.src.tar.xz
        popd
    done
}

while [ $# -gt 0 ]; do
    case $1 in
        -release | --release )
            shift
            release=$1
            ;;
        -rc | --rc )
            shift
            rc="rc$1"
            ;;
        -final | --final )
            rc="final"
            ;;
        -h | -help | --help )
            usage
            exit 0
            ;;
        * )
            echo "unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if [ "x$release" = "x" ]; then
    echo "error: need to specify a release version"
    exit 1
fi

# Make sure umask is not overly restrictive.
umask 0022

export_sources
exit 0
