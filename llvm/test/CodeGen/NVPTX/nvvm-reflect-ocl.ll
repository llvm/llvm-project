; Verify that __nvvm_reflect_ocl() is replaced with an appropriate value
;
; RUN: opt %s -S -passes='nvvm-reflect' -mtriple=nvptx64 -mcpu=sm_20 \
; RUN:   | FileCheck %s --check-prefixes=COMMON,SM20
; RUN: opt %s -S -passes='nvvm-reflect' -mtriple=nvptx64 -mcpu=sm_35 \
; RUN:   | FileCheck %s --check-prefixes=COMMON,SM35

@"$str" = private addrspace(4) constant [12 x i8] c"__CUDA_ARCH\00"

declare i32 @__nvvm_reflect_ocl(ptr addrspace(4) noundef)

; COMMON-LABEL: @foo
define i32 @foo(float %a, float %b) {
; COMMON-NOT: call i32 @__nvvm_reflect_ocl
  %reflect = tail call i32 @__nvvm_reflect_ocl(ptr addrspace(4) noundef @"$str")
; SM20: ret i32 200
; SM35: ret i32 350
  ret i32 %reflect
}

