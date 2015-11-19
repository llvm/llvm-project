; RUN: llc -mtriple thumbv7--windows-itanium -filetype asm -o - %s | FileCheck %s

@source = common global [512 x i8] zeroinitializer, align 4

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind

define void @function() {
entry:
  call void @llvm.memset.p0i8.i32(i8* bitcast ([512 x i8]* @source to i8*), i8 0, i32 512, i1 false)
  unreachable
}

; CHECK: movw r0, :lower16:source
; CHECK: movt r0, :upper16:source
; CHECK: movs r1, #0
; CHECK: mov.w r2, #512
; CHECK: memset

