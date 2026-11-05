; Test that the Global Registry internal's state is consistent even in the presence of re-use of memory allocations / addresses.
; It is a bit awkward to reproduce and only triggers with .bc inputs, and not .ll ones, hence the binary input.
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown -filetype=asm %S/Inputs/keep-global-registry-consistent.bc -o - | FileCheck %s
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown -filetype=asm %S/Inputs/keep-global-registry-consistent.bc -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %S/Inputs/keep-global-registry-consistent.bc -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %S/Inputs/keep-global-registry-consistent.bc -o - -filetype=obj | spirv-val %}

; CHECK: OpCapability Kernel
; CHECK: OpCapability Addresses
; CHECK: OpCapability Int8
; CHECK: OpCapability GenericPointer
; CHECK: OpCapability Linkage
; CHECK: OpCapability Int64
