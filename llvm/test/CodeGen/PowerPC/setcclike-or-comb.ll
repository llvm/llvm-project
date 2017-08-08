; RUN: llc -O0 -fast-isel=0 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

@a = external global i32, align 4
@b = external global i32, align 4

; Function Attrs: nounwind
define void @fn1() #0 {
entry:
  %0 = load i32, i32* @a, align 4
  %cmp = icmp ne i32 %0, 1
  %conv = zext i1 %cmp to i32
  %1 = load i32, i32* @b, align 4
  %cmp1 = icmp ne i32 0, %1
  %conv2 = zext i1 %cmp1 to i32
  %or = or i32 %conv, %conv2
  %xor = xor i32 1, %or
  %call = call signext i32 @fn2(i32 signext %xor)
  %conv4 = zext i1 undef to i32
  store i32 %conv4, i32* @b, align 4
  ret void

; CHECK-LABEL: @fn1
; CHECK: blr
}

declare signext i32 @fn2(i32 signext)

attributes #0 = { nounwind "target-cpu"="ppc64le" }

