; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430---elf"

declare void @llvm.va_start(ptr) nounwind
declare void @llvm.va_end(ptr) nounwind
declare void @llvm.va_copy(ptr, ptr) nounwind

define void @va_start(i16 %a, ...) nounwind {
entry:
; CHECK-LABEL: va_start:
; CHECK: sub #2, r1
  %vl = alloca ptr, align 2
  %vl1 = bitcast ptr %vl to ptr
; CHECK-NEXT: mov r1, [[REG:r[0-9]+]]
; CHECK-NEXT: add #6, [[REG]]
; CHECK-NEXT: mov [[REG]], 0(r1)
  call void @llvm.va_start(ptr %vl1)
  call void @llvm.va_end(ptr %vl1)
  ret void
}

define i16 @va_arg(ptr %vl) nounwind {
entry:
; CHECK-LABEL: va_arg:
  %vl.addr = alloca ptr, align 2
  store ptr %vl, ptr %vl.addr, align 2
; CHECK: mov r12, [[REG:r[0-9]+]]
; CHECK-NEXT: incd [[REG]]
; CHECK-NEXT: mov [[REG]], 0(r1)
  %0 = va_arg ptr %vl.addr, i16
; CHECK-NEXT: mov 0(r12), r12
  ret i16 %0
}

define void @va_copy(ptr %vl) nounwind {
entry:
; CHECK-LABEL: va_copy:
  %vl.addr = alloca ptr, align 2
  %vl2 = alloca ptr, align 2
; CHECK-DAG: mov r12, 2(r1)
  store ptr %vl, ptr %vl.addr, align 2
  %0 = bitcast ptr %vl2 to ptr
  %1 = bitcast ptr %vl.addr to ptr
; CHECK-DAG: mov r12, 0(r1)
  call void @llvm.va_copy(ptr %0, ptr %1)
  ret void
}
