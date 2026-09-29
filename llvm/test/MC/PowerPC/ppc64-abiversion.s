
# RUN: llvm-mc -triple powerpc64-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -h - | FileCheck %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown -filetype=obj %s | \
# RUN: llvm-readobj -h - | FileCheck %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown -filetype=null %s

	.abiversion 2
# CHECK: Flags [ (0x2)

