; RUN: llc -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430---elf"

declare void @llvm.va_start(i8*) nounwind
declare void @llvm.va_end(i8*) nounwind
declare void @llvm.va_copy(i8*, i8*) nounwind

define void @va_start(i16 %a, ...) nounwind {
entry:
; CHECK-LABEL: va_start:
; CHECK: sub.w #2, r1
  %vl = alloca i8*, align 2
  %vl1 = bitcast i8** %vl to i8*
; CHECK-NEXT: mov.w r1, [[REG:r[0-9]+]]
; CHECK-NEXT: add.w #6, [[REG]]
; CHECK-NEXT: mov.w [[REG]], 0(r1)
  call void @llvm.va_start(i8* %vl1)
  call void @llvm.va_end(i8* %vl1)
  ret void
}

define i16 @va_arg(i8* %vl) nounwind {
entry:
; CHECK-LABEL: va_arg:
  %vl.addr = alloca i8*, align 2
  store i8* %vl, i8** %vl.addr, align 2
; CHECK: mov.w r12, [[REG:r[0-9]+]]
; CHECK-NEXT: add.w #2, [[REG]]
; CHECK-NEXT: mov.w [[REG]], 0(r1)
  %0 = va_arg i8** %vl.addr, i16
; CHECK-NEXT: mov.w 0(r12), r12
  ret i16 %0
}

define void @va_copy(i8* %vl) nounwind {
entry:
; CHECK-LABEL: va_copy:
  %vl.addr = alloca i8*, align 2
  %vl2 = alloca i8*, align 2
; CHECK: mov.w r12, 2(r1)
  store i8* %vl, i8** %vl.addr, align 2
  %0 = bitcast i8** %vl2 to i8*
  %1 = bitcast i8** %vl.addr to i8*
; CHECK-NEXT: mov.w r12, 0(r1)
  call void @llvm.va_copy(i8* %0, i8* %1)
  ret void
}
