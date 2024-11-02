; RUN: llc -mtriple=thumbv7-windows-itanium -mcpu=cortex-a9 -verify-machineinstrs -o - %s | FileCheck %s

declare void @llvm.memmove.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind

@source = common global [512 x i8] zeroinitializer, align 4
@target = common global [512 x i8] zeroinitializer, align 4

define void @move() nounwind {
entry:
  call void @llvm.memmove.p0.p0.i32(ptr @target, ptr @source, i32 512, i1 false)
  unreachable
}

; CHECK-NOT: __aeabi_memmove

define void @copy() nounwind {
entry:
  call void @llvm.memcpy.p0.p0.i32(ptr @target, ptr @source, i32 512, i1 false)
  unreachable
}

; CHECK-NOT: __aeabi_memcpy

define i32 @divide(i32 %i, i32 %j) nounwind {
entry:
  %quotient = sdiv i32 %i, %j
  ret i32 %quotient
}

; CHECK-NOT: __aeabi_idiv

