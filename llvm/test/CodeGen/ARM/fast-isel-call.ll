; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -arm-long-calls | FileCheck %s --check-prefix=ARM-LONG
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -arm-long-calls | FileCheck %s --check-prefix=THUMB-LONG
; RUN: llc < %s -O0 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -mattr=-vfp2 | FileCheck %s --check-prefix=ARM-NOVFP
; RUN: llc < %s -O0 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -mattr=-vfp2 | FileCheck %s --check-prefix=THUMB-NOVFP

define i32 @t0(i1 zeroext %a) nounwind {
  %1 = zext i1 %a to i32
  ret i32 %1
}

define i32 @t1(i8 signext %a) nounwind {
  %1 = sext i8 %a to i32
  ret i32 %1
}

define i32 @t2(i8 zeroext %a) nounwind {
  %1 = zext i8 %a to i32
  ret i32 %1
}

define i32 @t3(i16 signext %a) nounwind {
  %1 = sext i16 %a to i32
  ret i32 %1
}

define i32 @t4(i16 zeroext %a) nounwind {
  %1 = zext i16 %a to i32
  ret i32 %1
}

define void @foo(i8 %a, i16 %b) nounwind {
; ARM: foo
; THUMB: foo
;; Materialize i1 1
; ARM: movw r2, #1
;; zero-ext
; ARM: and r2, r2, #1
; THUMB: and r2, r2, #1
  %1 = call i32 @t0(i1 zeroext 1)
; ARM: sxtb	r2, r1
; ARM: mov r0, r2
; THUMB: sxtb	r2, r1
; THUMB: mov r0, r2
  %2 = call i32 @t1(i8 signext %a)
; ARM: uxtb	r2, r1
; ARM: mov r0, r2
; THUMB: uxtb	r2, r1
; THUMB: mov r0, r2
  %3 = call i32 @t2(i8 zeroext %a)
; ARM: sxth	r2, r1
; ARM: mov r0, r2
; THUMB: sxth	r2, r1
; THUMB: mov r0, r2
  %4 = call i32 @t3(i16 signext %b)
; ARM: uxth	r2, r1
; ARM: mov r0, r2
; THUMB: uxth	r2, r1
; THUMB: mov r0, r2
  %5 = call i32 @t4(i16 zeroext %b)

;; A few test to check materialization
;; Note: i1 1 was materialized with t1 call
; ARM: movw r1, #255
%6 = call i32 @t2(i8 zeroext 255)
; ARM: movw r1, #65535
; THUMB: movw r1, #65535
%7 = call i32 @t4(i16 zeroext 65535)
  ret void
}

define void @foo2() nounwind {
  %1 = call signext i16 @t5()
  %2 = call zeroext i16 @t6()
  %3 = call signext i8 @t7()
  %4 = call zeroext i8 @t8()
  %5 = call zeroext i1 @t9()
  ret void
}

declare signext i16 @t5();
declare zeroext i16 @t6();
declare signext i8 @t7();
declare zeroext i8 @t8();
declare zeroext i1 @t9();

define i32 @t10(i32 %argc, i8** nocapture %argv) {
entry:
; ARM: @t10
; ARM: movw r0, #0
; ARM: movw r1, #248
; ARM: movw r2, #187
; ARM: movw r3, #28
; ARM: movw r9, #40
; ARM: movw r12, #186
; ARM: uxtb r0, r0
; ARM: uxtb r1, r1
; ARM: uxtb r2, r2
; ARM: uxtb r3, r3
; ARM: uxtb r9, r9
; ARM: str r9, [sp]
; ARM: uxtb r9, r12
; ARM: str r9, [sp, #4]
; ARM: bl _bar
; ARM-LONG: @t10
; ARM-LONG: movw lr, :lower16:L_bar$non_lazy_ptr
; ARM-LONG: movt lr, :upper16:L_bar$non_lazy_ptr
; ARM-LONG: ldr lr, [lr]
; ARM-LONG: blx lr
; THUMB: @t10
; THUMB: movs r0, #0
; THUMB: movt r0, #0
; THUMB: movs r1, #248
; THUMB: movt r1, #0
; THUMB: movs r2, #187
; THUMB: movt r2, #0
; THUMB: movs r3, #28
; THUMB: movt r3, #0
; THUMB: movw r9, #40
; THUMB: movt r9, #0
; THUMB: movw r12, #186
; THUMB: movt r12, #0
; THUMB: uxtb r0, r0
; THUMB: uxtb r1, r1
; THUMB: uxtb r2, r2
; THUMB: uxtb r3, r3
; THUMB: uxtb.w r9, r9
; THUMB: str.w r9, [sp]
; THUMB: uxtb.w r9, r12
; THUMB: str.w r9, [sp, #4]
; THUMB: bl _bar
; THUMB-LONG: @t10
; THUMB-LONG: movw lr, :lower16:L_bar$non_lazy_ptr
; THUMB-LONG: movt lr, :upper16:L_bar$non_lazy_ptr
; THUMB-LONG: ldr.w lr, [lr]
; THUMB-LONG: blx lr
  %call = call i32 @bar(i8 zeroext 0, i8 zeroext -8, i8 zeroext -69, i8 zeroext 28, i8 zeroext 40, i8 zeroext -70)
  ret i32 0
}

declare i32 @bar(i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext, i8 zeroext)

define i32 @bar0(i32 %i) nounwind {
  ret i32 0
}

define void @foo3() uwtable {
; ARM: movw    r0, #0
; ARM: movw    r1, :lower16:_bar0
; ARM: movt    r1, :upper16:_bar0
; ARM: blx     r1
; THUMB: movs    r0, #0
; THUMB: movw    r1, :lower16:_bar0
; THUMB: movt    r1, :upper16:_bar0
; THUMB: blx     r1
  %fptr = alloca i32 (i32)*, align 8
  store i32 (i32)* @bar0, i32 (i32)** %fptr, align 8
  %1 = load i32 (i32)** %fptr, align 8
  %call = call i32 %1(i32 0)
  ret void
}

define i32 @LibCall(i32 %a, i32 %b) {
entry:
; ARM: LibCall
; ARM: bl ___udivsi3
; ARM-LONG: LibCall
; ARM-LONG: movw r2, :lower16:L___udivsi3$non_lazy_ptr
; ARM-LONG: movt r2, :upper16:L___udivsi3$non_lazy_ptr
; ARM-LONG: ldr r2, [r2]
; ARM-LONG: blx r2
; THUMB: LibCall
; THUMB: bl ___udivsi3
; THUMB-LONG: LibCall
; THUMB-LONG: movw r2, :lower16:L___udivsi3$non_lazy_ptr
; THUMB-LONG: movt r2, :upper16:L___udivsi3$non_lazy_ptr
; THUMB-LONG: ldr r2, [r2]
; THUMB-LONG: blx r2
        %tmp1 = udiv i32 %a, %b         ; <i32> [#uses=1]
        ret i32 %tmp1
}

define i32 @VarArg() nounwind {
entry:
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  %m = alloca i32, align 4
  %n = alloca i32, align 4
  %tmp = alloca i32, align 4
  %0 = load i32* %i, align 4
  %1 = load i32* %j, align 4
  %2 = load i32* %k, align 4
  %3 = load i32* %m, align 4
  %4 = load i32* %n, align 4
; ARM: VarArg
; ARM: mov r7, sp
; ARM: movw r0, #5
; ARM: ldr r1, [r7, #-4]
; ARM: ldr r2, [r7, #-8]
; ARM: ldr r3, [r7, #-12]
; ARM: ldr r9, [sp, #16]
; ARM: ldr r12, [sp, #12]
; ARM: str r9, [sp]
; ARM: str r12, [sp, #4]
; ARM: bl _CallVariadic
; THUMB: mov r7, sp
; THUMB: movs r0, #5
; THUMB: movt r0, #0
; THUMB: ldr r1, [sp, #28]
; THUMB: ldr r2, [sp, #24]
; THUMB: ldr r3, [sp, #20]
; THUMB: ldr.w r9, [sp, #16]
; THUMB: ldr.w r12, [sp, #12]
; THUMB: str.w r9, [sp]
; THUMB: str.w r12, [sp, #4]
; THUMB: bl _CallVariadic
  %call = call i32 (i32, ...)* @CallVariadic(i32 5, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4)
  store i32 %call, i32* %tmp, align 4
  %5 = load i32* %tmp, align 4
  ret i32 %5
}

declare i32 @CallVariadic(i32, ...)

; Test fastcc

define fastcc void @fast_callee(float %i) ssp {
entry:
; ARM: fast_callee
; ARM: vmov r0, s0
; THUMB: fast_callee
; THUMB: vmov r0, s0
; ARM-NOVFP: fast_callee
; ARM-NOVFP-NOT: s0
; THUMB-NOVFP: fast_callee
; THUMB-NOVFP-NOT: s0
  call void @print(float %i)
  ret void
}

define void @fast_caller() ssp {
entry:
; ARM: fast_caller
; ARM: vldr s0,
; THUMB: fast_caller
; THUMB: vldr s0,
; ARM-NOVFP: fast_caller
; ARM-NOVFP: movw r0, #13107
; ARM-NOVFP: movt r0, #16611
; THUMB-NOVFP: fast_caller
; THUMB-NOVFP: movw r0, #13107
; THUMB-NOVFP: movt r0, #16611
  call fastcc void @fast_callee(float 0x401C666660000000)
  ret void
}

define void @no_fast_callee(float %i) ssp {
entry:
; ARM: no_fast_callee
; ARM: vmov s0, r0
; THUMB: no_fast_callee
; THUMB: vmov s0, r0
; ARM-NOVFP: no_fast_callee
; ARM-NOVFP-NOT: s0
; THUMB-NOVFP: no_fast_callee
; THUMB-NOVFP-NOT: s0
  call void @print(float %i)
  ret void
}

define void @no_fast_caller() ssp {
entry:
; ARM: no_fast_caller
; ARM: vmov r0, s0
; THUMB: no_fast_caller
; THUMB: vmov r0, s0
; ARM-NOVFP: no_fast_caller
; ARM-NOVFP: movw r0, #13107
; ARM-NOVFP: movt r0, #16611
; THUMB-NOVFP: no_fast_caller
; THUMB-NOVFP: movw r0, #13107
; THUMB-NOVFP: movt r0, #16611
  call void @no_fast_callee(float 0x401C666660000000)
  ret void
}

declare void @print(float)
