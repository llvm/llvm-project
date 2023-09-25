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
llvm_targets=$($llvm_config --targets-built)

for target in $llvm_targets; do
    touch "llvm_${target}.opam"
    mkdir -p backends/$target
    sed -e "s/@TARGET@/$target/g" \
        -e "s/@CFLAGS@/-DTARGET=$target/g" "backends/dune.in" > backends/$target/dune
    sed "s/@TARGET@/$target/g" "backends/llvm_backend.mli.in" > backends/$target/llvm_${target}.mli
    sed "s/@TARGET@/$target/g" "backends/llvm_backend.ml.in" > backends/$target/llvm_${target}.ml
    sed "s/@TARGET@/$target/g" "backends/backend_ocaml.c" > backends/$target/${target}_ocaml.c
done
sed "s/@LLVM_TARGETS_TO_BUILD@/$llvm_targets/g" "all_backends/dune.in" > all_backends/dune

append_context() {
    linking_mode=$1
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
    $llvm_config --link-static
    if [ $? -ne 0 ]; then
        echo "Static mode is not supported."
        exit 1
    fi
    append_context --link-static
fi
if [ $mode = "shared" ]; then
    $llvm_config --link-shared
    if [ $? -ne 0 ]; then
        echo "Shared mode is not supported."
        exit 1
    fi
    append_context --link-shared
fi
