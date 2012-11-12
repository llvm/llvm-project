; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -arm-long-calls | FileCheck %s --check-prefix=ARM-LONG
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -arm-long-calls | FileCheck %s --check-prefix=THUMB-LONG

@message1 = global [60 x i8] c"The LLVM Compiler Infrastructure\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00\00", align 1
@temp = common global [60 x i8] zeroinitializer, align 1

define void @t1() nounwind ssp {
; ARM: t1
; ARM: movw r0, :lower16:_message1
; ARM: movt r0, :upper16:_message1
; ARM: add r0, r0, #5
; ARM: movw r1, #64
; ARM: movw r2, #10
; ARM: uxtb r1, r1
; ARM: bl _memset
; ARM-LONG: t1
; ARM-LONG: movw r3, :lower16:L_memset$non_lazy_ptr
; ARM-LONG: movt r3, :upper16:L_memset$non_lazy_ptr
; ARM-LONG: ldr r3, [r3]
; ARM-LONG: blx r3
; THUMB: t1
; THUMB: movw r0, :lower16:_message1
; THUMB: movt r0, :upper16:_message1
; THUMB: adds r0, #5
; THUMB: movs r1, #64
; THUMB: movt r1, #0
; THUMB: movs r2, #10
; THUMB: movt r2, #0
; THUMB: uxtb r1, r1
; THUMB: bl _memset
; THUMB-LONG: t1
; THUMB-LONG: movw r3, :lower16:L_memset$non_lazy_ptr
; THUMB-LONG: movt r3, :upper16:L_memset$non_lazy_ptr
; THUMB-LONG: ldr r3, [r3]
; THUMB-LONG: blx r3
  call void @llvm.memset.p0i8.i32(i8* getelementptr inbounds ([60 x i8]* @message1, i32 0, i32 5), i8 64, i32 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define void @t2() nounwind ssp {
; ARM: t2
; ARM: movw r0, :lower16:L_temp$non_lazy_ptr
; ARM: movt r0, :upper16:L_temp$non_lazy_ptr
; ARM: ldr r0, [r0]
; ARM: add r1, r0, #4
; ARM: add r0, r0, #16
; ARM: movw r2, #17
; ARM: str r0, [sp]                @ 4-byte Spill
; ARM: mov r0, r1
; ARM: ldr r1, [sp]                @ 4-byte Reload
; ARM: bl _memcpy
; ARM-LONG: t2
; ARM-LONG: movw r3, :lower16:L_memcpy$non_lazy_ptr
; ARM-LONG: movt r3, :upper16:L_memcpy$non_lazy_ptr
; ARM-LONG: ldr r3, [r3]
; ARM-LONG: blx r3
; THUMB: t2
; THUMB: movw r0, :lower16:L_temp$non_lazy_ptr
; THUMB: movt r0, :upper16:L_temp$non_lazy_ptr
; THUMB: ldr r0, [r0]
; THUMB: adds r1, r0, #4
; THUMB: adds r0, #16
; THUMB: movs r2, #17
; THUMB: movt r2, #0
; THUMB: mov r0, r1
; THUMB: bl _memcpy
; THUMB-LONG: t2
; THUMB-LONG: movw r3, :lower16:L_memcpy$non_lazy_ptr
; THUMB-LONG: movt r3, :upper16:L_memcpy$non_lazy_ptr
; THUMB-LONG: ldr r3, [r3]
; THUMB-LONG: blx r3
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds ([60 x i8]* @temp, i32 0, i32 4), i8* getelementptr inbounds ([60 x i8]* @temp, i32 0, i32 16), i32 17, i32 1, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind

define void @t3() nounwind ssp {
; ARM: t3
; ARM: movw r0, :lower16:L_temp$non_lazy_ptr
; ARM: movt r0, :upper16:L_temp$non_lazy_ptr
; ARM: ldr r0, [r0]
; ARM: add r1, r0, #4
; ARM: add r0, r0, #16
; ARM: movw r2, #10
; ARM: mov r0, r1
; ARM: bl _memmove
; ARM-LONG: t3
; ARM-LONG: movw r3, :lower16:L_memmove$non_lazy_ptr
; ARM-LONG: movt r3, :upper16:L_memmove$non_lazy_ptr
; ARM-LONG: ldr r3, [r3]
; ARM-LONG: blx r3
; THUMB: t3
; THUMB: movw r0, :lower16:L_temp$non_lazy_ptr
; THUMB: movt r0, :upper16:L_temp$non_lazy_ptr
; THUMB: ldr r0, [r0]
; THUMB: adds r1, r0, #4
; THUMB: adds r0, #16
; THUMB: movs r2, #10
; THUMB: movt r2, #0
; THUMB: mov r0, r1
; THUMB: bl _memmove
; THUMB-LONG: t3
; THUMB-LONG: movw r3, :lower16:L_memmove$non_lazy_ptr
; THUMB-LONG: movt r3, :upper16:L_memmove$non_lazy_ptr
; THUMB-LONG: ldr r3, [r3]
; THUMB-LONG: blx r3
  call void @llvm.memmove.p0i8.p0i8.i32(i8* getelementptr inbounds ([60 x i8]* @temp, i32 0, i32 4), i8* getelementptr inbounds ([60 x i8]* @temp, i32 0, i32 16), i32 10, i32 1, i1 false)
  ret void
}

define void @t4() nounwind ssp {
; ARM: t4
; ARM: movw r0, :lower16:L_temp$non_lazy_ptr
; ARM: movt r0, :upper16:L_temp$non_lazy_ptr
; ARM: ldr r0, [r0]
; ARM: ldr r1, [r0, #16]
; ARM: str r1, [r0, #4]
; ARM: ldr r1, [r0, #20]
; ARM: str r1, [r0, #8]
; ARM: ldrh r1, [r0, #24]
; ARM: strh r1, [r0, #12]
; ARM: bx lr
; THUMB: movw r0, :lower16:L_temp$non_lazy_ptr
; THUMB: movt r0, :upper16:L_temp$non_lazy_ptr
; THUMB: ldr r0, [r0]
; THUMB: ldr r1, [r0, #16]
; THUMB: str r1, [r0, #4]
; THUMB: ldr r1, [r0, #20]
; THUMB: str r1, [r0, #8]
; THUMB: ldrh r1, [r0, #24]
; THUMB: strh r1, [r0, #12]
; THUMB: bx lr
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds ([60 x i8]* @temp, i32 0, i32 4), i8* getelementptr inbounds ([60 x i8]* @temp, i32 0, i32 16), i32 10, i32 1, i1 false)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
