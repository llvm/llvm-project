; -spirv-preserve-auxdata is an error unless SPV_KHR_non_semantic_info is on.
; AMD triples behave as if -spirv-preserve-auxdata were passed.

; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown \
; RUN:   -spirv-preserve-auxdata %s -o - 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-amd-amdhsa \
; RUN:   --spirv-ext=-SPV_KHR_non_semantic_info %s -o - 2>&1 | FileCheck %s

; CHECK: -spirv-preserve-auxdata requires the SPV_KHR_non_semantic_info extension

target triple = "spir64-unknown-unknown"

define spir_func void @fn() "my-attr"="val" {
  ret void
}
