#!/bin/sh

set -e
set -x

if test ! "$(dirname $0)" -ef '.'; then
    echo "The script must be executed from its current directory."
    exit 1
fi

if test "$#" -ne 2; then
    echo "Usage: $0 <llvm-config> <linking mode>"
    exit 1
fi

llvm_config=$1
mode=$2

base_cflags=$($llvm_config --cflags)
ldflags="$($llvm_config --ldflags) -lstdc++ -fPIC"
llvm_targets=$($llvm_config --targets-built)

append_context() {
    context_name=$1
    linking_mode=$2
    echo "(context (default
 (env
  (_
   (ocamlc_flags -custom)
   (c_flags $base_cflags)
   (env-vars
    (LLVM_CONFIG $llvm_config)
    (LINK_MODE $linking_mode))))))
" >> "dune-workspace"
}

echo "(lang dune 3.2)
" > "dune-workspace"

if [ $mode = "static" ]; then
    $llvm_config --link-static --libs
    if [ $? -ne 0 ]; then
        echo "Static mode is not supported."
        exit 1
    fi
    append_context static --link-static
fi
if [ $mode = "shared" ]; then
    $llvm_config --link-shared --libs
    if [ $? -ne 0 ]; then
        echo "Shared mode is not supported."
        exit 1
    fi
    append_context shared --link-shared
fi
