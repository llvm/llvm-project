; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -verify-machineinstrs | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=THUMB
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -mattr=+long-calls -verify-machineinstrs | FileCheck %s --check-prefix=ARM-LONG
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -mattr=+long-calls -verify-machineinstrs | FileCheck %s --check-prefix=ARM-LONG
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -mattr=+long-calls -verify-machineinstrs | FileCheck %s --check-prefix=THUMB-LONG

; Note that some of these tests assume that relocations are either
; movw/movt or constant pool loads. Different platforms will select
; different approaches.

@message1 = global [60 x i8] c"The LLVM Compiler Infrastructure\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 1
@temp = common global [60 x i8] zeroinitializer, align 1

define void @t1() nounwind ssp {
; ARM-LABEL: t1:
; ARM: {{(movw r0, :lower16:_?message1)|(ldr r0, .LCPI)}}
; ARM: {{(movt r0, :upper16:_?message1)|(ldr r0, \[r0\])}}
; ARM: add r0, r0, #5
; ARM: movw r1, #64
; ARM: movw r2, #10
; ARM: and r1, r1, #255
; ARM: bl {{_?}}memset
; ARM-LONG-LABEL: t1:
; ARM-LONG: {{(movw r3, :lower16:L_memset\$non_lazy_ptr)|(ldr r3, .LCPI)}}
; ARM-LONG: {{(movt r3, :upper16:L_memset\$non_lazy_ptr)?}}
; ARM-LONG: ldr r3, [r3]
; ARM-LONG: blx r3
; THUMB-LABEL: t1:
; THUMB: {{(movw r0, :lower16:_?message1)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:_?message1)|(ldr r0, \[r0\])}}
; THUMB: adds r0, #5
; THUMB: movs r1, #64
; THUMB: movs r2, #10
; THUMB: and r1, r1, #255
; THUMB: bl {{_?}}memset
; THUMB-LONG-LABEL: t1:
; THUMB-LONG: movw r3, :lower16:L_memset$non_lazy_ptr
; THUMB-LONG: movt r3, :upper16:L_memset$non_lazy_ptr
; THUMB-LONG: ldr r3, [r3]
; THUMB-LONG: blx r3
  call void @llvm.memset.p0i8.i32(i8* align 4 getelementptr inbounds ([60 x i8], [60 x i8]* @message1, i32 0, i32 5), i8 64, i32 10, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind

define void @t2() nounwind ssp {
; ARM-LABEL: t2:
; ARM: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; ARM: ldr r0, [r0]
; ARM: add r1, r0, #4
; ARM: add r0, r0, #16
; ARM: movw r2, #17
; ARM: str r0, [sp[[SLOT:[, #0-9]*]]] @ 4-byte Spill
; ARM: mov r0, r1
; ARM: ldr r1, [sp[[SLOT]]] @ 4-byte Reload
; ARM: bl {{_?}}memcpy
; ARM-LONG-LABEL: t2:
; ARM-LONG: {{(movw r3, :lower16:L_memcpy\$non_lazy_ptr)|(ldr r3, .LCPI)}}
; ARM-LONG: {{(movt r3, :upper16:L_memcpy\$non_lazy_ptr)?}}
; ARM-LONG: ldr r3, [r3]
; ARM-LONG: blx r3
; THUMB-LABEL: t2:
; THUMB: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; THUMB: ldr r0, [r0]
; THUMB: adds r1, r0, #4
; THUMB: adds r0, #16
; THUMB: movs r2, #17
; THUMB: str r0, [sp[[SLOT:[, #0-9]*]]] @ 4-byte Spill
; THUMB: mov r0, r1
; THUMB: ldr r1,  [sp[[SLOT]]] @ 4-byte Reload
; THUMB: bl {{_?}}memcpy
; THUMB-LONG-LABEL: t2:
; THUMB-LONG: movw r3, :lower16:L_memcpy$non_lazy_ptr
; THUMB-LONG: movt r3, :upper16:L_memcpy$non_lazy_ptr
; THUMB-LONG: ldr r3, [r3]
; THUMB-LONG: blx r3
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* align 4 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 17, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

define void @t3() nounwind ssp {
; ARM-LABEL: t3:
; ARM: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; ARM: ldr r0, [r0]
; ARM: add r1, r0, #4
; ARM: add r0, r0, #16
; ARM: movw r2, #10
; ARM: mov r0, r1
; ARM: bl {{_?}}memmove
; ARM-LONG-LABEL: t3:
; ARM-LONG: {{(movw r3, :lower16:L_memmove\$non_lazy_ptr)|(ldr r3, .LCPI)}}
; ARM-LONG: {{(movt r3, :upper16:L_memmove\$non_lazy_ptr)?}}
; ARM-LONG: ldr r3, [r3]
; ARM-LONG: blx r3
; THUMB-LABEL: t3:
; THUMB: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; THUMB: ldr r0, [r0]
; THUMB: adds r1, r0, #4
; THUMB: adds r0, #16
; THUMB: movs r2, #10
; THUMB: str r0, [sp[[SLOT:[, #0-9]*]]] @ 4-byte Spill
; THUMB: mov r0, r1
; THUMB: ldr r1,  [sp[[SLOT]]] @ 4-byte Reload
; THUMB: bl {{_?}}memmove
; THUMB-LONG-LABEL: t3:
; THUMB-LONG: movw r3, :lower16:L_memmove$non_lazy_ptr
; THUMB-LONG: movt r3, :upper16:L_memmove$non_lazy_ptr
; THUMB-LONG: ldr r3, [r3]
; THUMB-LONG: blx r3
  call void @llvm.memmove.p0i8.p0i8.i32(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 10, i1 false)
  ret void
}

define void @t4() nounwind ssp {
; ARM-LABEL: t4:
; ARM: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; ARM: ldr r0, [r0]
; ARM: ldr r1, [r0, #16]
; ARM: str r1, [r0, #4]
; ARM: ldr r1, [r0, #20]
; ARM: str r1, [r0, #8]
; ARM: ldrh r1, [r0, #24]
; ARM: strh r1, [r0, #12]
; ARM: bx lr
; THUMB-LABEL: t4:
; THUMB: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; THUMB: ldr r0, [r0]
; THUMB: ldr r1, [r0, #16]
; THUMB: str r1, [r0, #4]
; THUMB: ldr r1, [r0, #20]
; THUMB: str r1, [r0, #8]
; THUMB: ldrh r1, [r0, #24]
; THUMB: strh r1, [r0, #12]
; THUMB: bx lr
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 10, i1 false)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

define void @t5() nounwind ssp {
; ARM-LABEL: t5:
; ARM: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; ARM: ldr r0, [r0]
; ARM: ldrh r1, [r0, #16]
; ARM: strh r1, [r0, #4]
; ARM: ldrh r1, [r0, #18]
; ARM: strh r1, [r0, #6]
; ARM: ldrh r1, [r0, #20]
; ARM: strh r1, [r0, #8]
; ARM: ldrh r1, [r0, #22]
; ARM: strh r1, [r0, #10]
; ARM: ldrh r1, [r0, #24]
; ARM: strh r1, [r0, #12]
; ARM: bx lr
; THUMB-LABEL: t5:
; THUMB: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; THUMB: ldr r0, [r0]
; THUMB: ldrh r1, [r0, #16]
; THUMB: strh r1, [r0, #4]
; THUMB: ldrh r1, [r0, #18]
; THUMB: strh r1, [r0, #6]
; THUMB: ldrh r1, [r0, #20]
; THUMB: strh r1, [r0, #8]
; THUMB: ldrh r1, [r0, #22]
; THUMB: strh r1, [r0, #10]
; THUMB: ldrh r1, [r0, #24]
; THUMB: strh r1, [r0, #12]
; THUMB: bx lr
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* align 2 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 10, i1 false)
  ret void
}

define void @t6() nounwind ssp {
; ARM-LABEL: t6:
; ARM: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr r0, .LCPI)}}
; ARM: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; ARM: ldr r0, [r0]
; ARM: ldrb r1, [r0, #16]
; ARM: strb r1, [r0, #4]
; ARM: ldrb r1, [r0, #17]
; ARM: strb r1, [r0, #5]
; ARM: ldrb r1, [r0, #18]
; ARM: strb r1, [r0, #6]
; ARM: ldrb r1, [r0, #19]
; ARM: strb r1, [r0, #7]
; ARM: ldrb r1, [r0, #20]
; ARM: strb r1, [r0, #8]
; ARM: ldrb r1, [r0, #21]
; ARM: strb r1, [r0, #9]
; ARM: ldrb r1, [r0, #22]
; ARM: strb r1, [r0, #10]
; ARM: ldrb r1, [r0, #23]
; ARM: strb r1, [r0, #11]
; ARM: ldrb r1, [r0, #24]
; ARM: strb r1, [r0, #12]
; ARM: ldrb r1, [r0, #25]
; ARM: strb r1, [r0, #13]
; ARM: bx lr
; THUMB-LABEL: t6:
; THUMB: {{(movw r0, :lower16:L_temp\$non_lazy_ptr)|(ldr.n r0, .LCPI)}}
; THUMB: {{(movt r0, :upper16:L_temp\$non_lazy_ptr)?}}
; THUMB: ldr r0, [r0]
; THUMB: ldrb r1, [r0, #16]
; THUMB: strb r1, [r0, #4]
; THUMB: ldrb r1, [r0, #17]
; THUMB: strb r1, [r0, #5]
; THUMB: ldrb r1, [r0, #18]
; THUMB: strb r1, [r0, #6]
; THUMB: ldrb r1, [r0, #19]
; THUMB: strb r1, [r0, #7]
; THUMB: ldrb r1, [r0, #20]
; THUMB: strb r1, [r0, #8]
; THUMB: ldrb r1, [r0, #21]
; THUMB: strb r1, [r0, #9]
; THUMB: ldrb r1, [r0, #22]
; THUMB: strb r1, [r0, #10]
; THUMB: ldrb r1, [r0, #23]
; THUMB: strb r1, [r0, #11]
; THUMB: ldrb r1, [r0, #24]
; THUMB: strb r1, [r0, #12]
; THUMB: ldrb r1, [r0, #25]
; THUMB: strb r1, [r0, #13]
; THUMB: bx lr
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* align 1 getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 10, i1 false)
  ret void
}

; rdar://13202135
define void @t7() nounwind ssp {
; Just make sure this doesn't assert when we have an odd length and an alignment of 2.
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 4), i8* getelementptr inbounds ([60 x i8], [60 x i8]* @temp, i32 0, i32 16), i32 3, i1 false)
  ret void
}

define i32 @t8(i32 %x) nounwind {
entry:
; ARM-LABEL: t8:
; ARM-NOT: FastISel missed call:   %expval = call i32 @llvm.expect.i32(i32 %x, i32 1)
; THUMB-LABEL: t8:
; THUMB-NOT: FastISel missed call:   %expval = call i32 @llvm.expect.i32(i32 %x, i32 1)
  %expval = call i32 @llvm.expect.i32(i32 %x, i32 1)
  ret i32 %expval
}

declare i32 @llvm.expect.i32(i32, i32) nounwind readnone
