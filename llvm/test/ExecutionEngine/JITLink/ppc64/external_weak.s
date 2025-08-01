# REQUIRES: asserts
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o \
# RUN:   %t/external_weak.o %S/Inputs/external_weak.s
# RUN: llvm-mc -triple=powerpc64le-unknown-linux-gnu -filetype=obj -o \
# RUN:   %t/external_weak_main.o %S/Inputs/external_weak_main.s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:              %t/external_weak.o %t/external_weak_main.o 2>&1 \
# RUN:              | FileCheck %s
# CHECK: Created ELFLinkGraphBuilder for "{{.*}}external_weak_main.o"
# CHECK: Creating defined graph symbol for ELF symbol "foo"
# CHECK: External symbols:
# CHECK:   {{.*}} linkage: weak, scope: default, dead  -   foo
# CHECK: section .text:
# CHECK:   {{.*}} kind = CallBranchDeltaRestoreTOC, target = addressable@{{.*}}
# `foo` is weak in both relocatable files. `foo` is resolved to the one
# defined in `%t/external_weak.o`. So calling `foo` in `%t/external_weak_main.o`
# is expected to be an external function call.
