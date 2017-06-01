; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:4"
target triple = "x86_64-unknown-linux-gnu"

define i8 addrspace(4)* @f_0() {
; CHECK-LABEL: @f_0(
; CHECK: ret i8 addrspace(4)* getelementptr (i8, i8 addrspace(4)* null, i64 50)
  %result = getelementptr i8, i8 addrspace(4)* null, i64 50
  ret i8 addrspace(4)* %result
}

define i8 addrspace(3)* @f_1() {
; inttoptr is fine here since addrspace(3) is integral.

; CHECK-LABEL: @f_1(
; CHECK: ret i8 addrspace(3)* inttoptr (i64 50 to i8 addrspace(3)*)
  %result = getelementptr i8, i8 addrspace(3)* null, i64 50
  ret i8 addrspace(3)* %result
}

define void @f_2(i8 addrspace(4)** %ptr0, i8 addrspace(4)** %ptr1) {
; It is not okay to convert the load/store pair to load and store
; integers, since pointers in address space 4 are non-integral.

; CHECK-LABEL: @f_2(
entry:
; CHECK:  %val = load i8 addrspace(4)*, i8 addrspace(4)** %ptr0, align 8
; CHECK:  store i8 addrspace(4)* %val, i8 addrspace(4)** %ptr1, align 8
; CHECK-NOT: load i64
; CHECK-NOT: store i64
  %val = load i8 addrspace(4)*, i8 addrspace(4)** %ptr0
  store i8 addrspace(4)* %val, i8 addrspace(4)** %ptr1
  ret void
}

define void @f_3(i8 addrspace(3)** %ptr0, i8 addrspace(3)** %ptr1) {
; It *is* okay to convert the load/store pair to load and store
; integers, since pointers in address space 3 are integral.

; CHECK-LABEL: @f_3(
entry:
; CHECK: load i64
; CHECK:  store i64
  %val = load i8 addrspace(3)*, i8 addrspace(3)** %ptr0
  store i8 addrspace(3)* %val, i8 addrspace(3)** %ptr1
  ret void
}
