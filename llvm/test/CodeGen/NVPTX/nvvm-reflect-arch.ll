; Libdevice in recent CUDA versions relies on __CUDA_ARCH reflecting GPU type.
; Verify that __nvvm_reflect() is replaced with an appropriate value.
;
; RUN: opt %s -S -passes='default<O2>' -mtriple=nvptx64 -mcpu=sm_20 \
; RUN:   | FileCheck %s --check-prefixes=COMMON,SM20
; RUN: opt %s -S -passes='default<O2>' -mtriple=nvptx64 -mcpu=sm_35 \
; RUN:   | FileCheck %s --check-prefixes=COMMON,SM35

@"$str" = private addrspace(1) constant [12 x i8] c"__CUDA_ARCH\00"

declare i32 @__nvvm_reflect(ptr)

; COMMON-LABEL: @foo
define i32 @foo(float %a, float %b) {
; COMMON-NOT: call i32 @__nvvm_reflect
  %reflect = call i32 @__nvvm_reflect(ptr addrspacecast (ptr addrspace(1) @"$str" to ptr))
; SM20: ret i32 200  
; SM35: ret i32 350  
  ret i32 %reflect
}
