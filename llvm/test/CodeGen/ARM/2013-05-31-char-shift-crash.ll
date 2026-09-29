; RUN: llc < %s -O0 -mtriple=armv4t--linux-eabi-android
; RUN: llc < %s -O0 -mtriple=armv4t-unknown-linux
; RUN: llc < %s -O0 -mtriple=armv5-unknown-linux

; See http://llvm.org/bugs/show_bug.cgi?id=16178
; ARMFastISel used to fail emitting sext/zext in pre-ARMv6.

; Function Attrs: nounwind
define arm_aapcscc void @f2(i8 signext %a) #0 {
entry:
  %a.addr = alloca i8, align 1
  store i8 %a, ptr %a.addr, align 1
  %0 = load i8, ptr %a.addr, align 1
  %conv = sext i8 %0 to i32
  %shr = ashr i32 %conv, 56
  %conv1 = trunc i32 %shr to i8
  call arm_aapcscc void @f1(i8 signext %conv1)
  ret void
}

declare arm_aapcscc void @f1(i8 signext) #1
