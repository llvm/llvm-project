; RUN: llc -O0 -mtriple=spirv64v1.3-unknown-unknown  %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.3-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; kernel void test(int global *in, int global *in2) {
;;   if (!in)
;;     return;
;;   if (in == 1)
;;     return;
;;   if (in > in2)
;;     return;
;;   if (in < in2)
;;     return;
;; }

; CHECK-SPIRV: OpConvertPtrToU
; CHECK-SPIRV: OpConvertPtrToU
; CHECK-SPIRV: OpINotEqual
; CHECK-SPIRV: OpConvertPtrToU
; CHECK-SPIRV: OpConvertPtrToU
; CHECK-SPIRV: OpIEqual
; CHECK-SPIRV: OpConvertPtrToU
; CHECK-SPIRV: OpConvertPtrToU
; CHECK-SPIRV: OpUGreaterThan
; CHECK-SPIRV: OpConvertPtrToU
; CHECK-SPIRV: OpConvertPtrToU
; CHECK-SPIRV: OpULessThan

define dso_local spir_kernel void @test(i32 addrspace(1)* noundef %in, i32 addrspace(1)* noundef %in2) {
entry:
  %in.addr = alloca i32 addrspace(1)*, align 8
  %in2.addr = alloca i32 addrspace(1)*, align 8
  store i32 addrspace(1)* %in, i32 addrspace(1)** %in.addr, align 8
  store i32 addrspace(1)* %in2, i32 addrspace(1)** %in2.addr, align 8
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %in.addr, align 8
  %tobool = icmp ne i32 addrspace(1)* %0, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %if.end8

if.end:                                           ; preds = %entry
  %1 = load i32 addrspace(1)*, i32 addrspace(1)** %in.addr, align 8
  %cmp = icmp eq i32 addrspace(1)* %1, inttoptr (i64 1 to i32 addrspace(1)*)
  br i1 %cmp, label %if.then1, label %if.end2

if.then1:                                         ; preds = %if.end
  br label %if.end8

if.end2:                                          ; preds = %if.end
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %in.addr, align 8
  %3 = load i32 addrspace(1)*, i32 addrspace(1)** %in2.addr, align 8
  %cmp3 = icmp ugt i32 addrspace(1)* %2, %3
  br i1 %cmp3, label %if.then4, label %if.end5

if.then4:                                         ; preds = %if.end2
  br label %if.end8

if.end5:                                          ; preds = %if.end2
  %4 = load i32 addrspace(1)*, i32 addrspace(1)** %in.addr, align 8
  %5 = load i32 addrspace(1)*, i32 addrspace(1)** %in2.addr, align 8
  %cmp6 = icmp ult i32 addrspace(1)* %4, %5
  br i1 %cmp6, label %if.then7, label %if.end8

if.then7:                                         ; preds = %if.end5
  br label %if.end8

if.end8:                                          ; preds = %if.then, %if.then1, %if.then4, %if.then7, %if.end5
  ret void
}
