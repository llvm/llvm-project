; RUN: llc -mtriple=thumbv7-apple-darwin-eabi < %s | FileCheck %s
; RUN: llc -mtriple=thumbv6m-apple-darwin-eabi -disable-fp-elim < %s | FileCheck %s --check-prefix=CHECK-T1
; RUN: llc -mtriple=thumbv7-apple-darwin-ios -disable-fp-elim < %s | FileCheck %s --check-prefix=CHECK-IOS


declare void @bar(i8*)

%bigVec = type [2 x double]

@var = global %bigVec zeroinitializer

define void @check_simple() minsize {
; CHECK-LABEL: check_simple:
; CHECK: push.w {r7, r8, r9, r10, r11, lr}
; CHECK-NOT: sub sp, sp,
; ...
; CHECK-NOT: add sp, sp,
; CHECK: pop.w {r0, r1, r2, r3, r11, pc}

; CHECK-T1-LABEL: check_simple:
; CHECK-T1: push {r3, r4, r5, r6, r7, lr}
; CHECK-T1: add r7, sp, #16
; CHECK-T1-NOT: sub sp, sp,
; ...
; CHECK-T1-NOT: add sp, sp,
; CHECK-T1: pop {r0, r1, r2, r3, r7, pc}

  ; iOS always has a frame pointer and messing with the push affects
  ; how it's set in the prologue. Make sure we get that right.
; CHECK-IOS-LABEL: check_simple:
; CHECK-IOS: push {r3, r4, r5, r6, r7, lr}
; CHECK-NOT: sub sp,
; CHECK-IOS: add r7, sp, #16
; CHECK-NOT: sub sp,
; ...
; CHECK-NOT: add sp,
; CHEC: pop {r3, r4, r5, r6, r7, pc}

  %var = alloca i8, i32 16
  call void @bar(i8* %var)
  ret void
}

define void @check_simple_too_big() minsize {
; CHECK-LABEL: check_simple_too_big:
; CHECK: push.w {r11, lr}
; CHECK: sub sp,
; ...
; CHECK: add sp,
; CHECK: pop.w {r11, pc}
  %var = alloca i8, i32 64
  call void @bar(i8* %var)
  ret void
}

define void @check_vfp_fold() minsize {
; CHECK-LABEL: check_vfp_fold:
; CHECK: push {r[[GLOBREG:[0-9]+]], lr}
; CHECK: vpush {d6, d7, d8, d9}
; CHECK-NOT: sub sp,
; ...
; CHECK: vldmia r[[GLOBREG]], {d8, d9}
; ...
; CHECK-NOT: add sp,
; CHECK: vpop {d6, d7, d8, d9}
; CHECKL pop {r[[GLOBREG]], pc}

  ; iOS uses aligned NEON stores here, which is convenient since we
  ; want to make sure that works too.
; CHECK-IOS-LABEL: check_vfp_fold:
; CHECK-IOS: push {r0, r1, r2, r3, r4, r7, lr}
; CHECK-IOS: sub.w r4, sp, #16
; CHECK-IOS: bic r4, r4, #15
; CHECK-IOS: mov sp, r4
; CHECK-IOS: vst1.64 {d8, d9}, [r4:128]
; ...
; CHECK-IOS: add r4, sp, #16
; CHECK-IOS: vld1.64 {d8, d9}, [r4:128]
; CHECK-IOS: mov sp, r4
; CHECK-IOS: pop {r4, r7, pc}

  %var = alloca i8, i32 16

  %tmp = load %bigVec* @var
  call void @bar(i8* %var)
  store %bigVec %tmp, %bigVec* @var

  ret void
}

; This function should use just enough space that the "add sp, sp, ..." could be
; folded in except that doing so would clobber the value being returned.
define i64 @check_no_return_clobber() minsize {
; CHECK-LABEL: check_no_return_clobber:
; CHECK: push.w {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, lr}
; CHECK-NOT: sub sp,
; ...
; CHECK: add sp, #40
; CHECK: pop.w {r11, pc}

  ; Just to keep iOS FileCheck within previous function:
; CHECK-IOS-LABEL: check_no_return_clobber:

  %var = alloca i8, i32 40
  call void @bar(i8* %var)
  ret i64 0
}

define arm_aapcs_vfpcc double @check_vfp_no_return_clobber() minsize {
; CHECK-LABEL: check_vfp_no_return_clobber:
; CHECK: push {r[[GLOBREG:[0-9]+]], lr}
; CHECK: vpush {d0, d1, d2, d3, d4, d5, d6, d7, d8, d9}
; CHECK-NOT: sub sp,
; ...
; CHECK: add sp, #64
; CHECK: vpop {d8, d9}
; CHECK: pop {r[[GLOBREG]], pc}

  %var = alloca i8, i32 64

  %tmp = load %bigVec* @var
  call void @bar(i8* %var)
  store %bigVec %tmp, %bigVec* @var

  ret double 1.0
}

@dbl = global double 0.0

; PR18136: there was a bug determining where the first eligible pop in a
; basic-block was when the entire block was epilogue code.
define void @test_fold_point(i1 %tst) minsize {
; CHECK-LABEL: test_fold_point:

  ; Important to check for beginning of basic block, because if it gets
  ; if-converted the test is probably no longer checking what it should.
; CHECK: {{LBB[0-9]+_2}}:
; CHECK-NEXT: vpop {d7, d8}
; CHECK-NEXT: pop {r4, pc}

  ; With a guaranteed frame-pointer, we want to make sure that its offset in the
  ; push block is correct, even if a few registers have been tacked onto a later
  ; vpush (PR18160).
; CHECK-IOS-LABEL: test_fold_point:
; CHECK-IOS: push {r4, r7, lr}
; CHECK-IOS-NEXT: add r7, sp, #4
; CHECK-IOS-NEXT: vpush {d7, d8}

  ; We want some memory so there's a stack adjustment to fold...
  %var = alloca i8, i32 8

  ; We want a long-lived floating register so that a callee-saved dN is used and
  ; there's both a vpop and a pop.
  %live_val = load double* @dbl
  br i1 %tst, label %true, label %end
true:
  call void @bar(i8* %var)
  store double %live_val, double* @dbl
  br label %end
end:
  ; We want the epilogue to be the only thing in a basic block so that we hit
  ; the correct edge-case (first inst in block is correct one to adjust).
  ret void
}