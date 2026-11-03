; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown --spv-allow-unknown-intrinsics=llvm %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv64-amd-amdhsa %s -o /dev/null 2>&1 | FileCheck %s

; An intrinsic that takes a metadata argument has no SPIR-V representation and
; can't be turned into a function call.

; CHECK: cannot lower the intrinsic 'llvm.type.test' that takes a metadata argument

define spir_func i1 @foo(ptr %p) {
  %r = call i1 @llvm.type.test(ptr %p, metadata !"typeid")
  ret i1 %r
}

declare i1 @llvm.type.test(ptr, metadata)
