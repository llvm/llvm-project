; RUN: llc < %s -mtriple=armv5te | FileCheck %s --check-prefix=V5
; RUN: llc < %s -mtriple=armv6   | FileCheck %s --check-prefix=V6
; RUN: llc < %s -mtriple=armv6t2 | FileCheck %s --check-prefix=V6T2
; RUN: llc < %s -mtriple=armv7   | FileCheck %s --check-prefix=V7
; PR18364

define i64 @f() #0 {
entry:
; V5-NOT: movw
; V6-NOT: movw
; V6T2: movw
; V7: movw
  %y = alloca i64, align 8
  %z = alloca i64, align 8
  store i64 1, ptr %y, align 8
  store i64 11579764786944, ptr %z, align 8
  %0 = load i64, ptr %y, align 8
  %1 = load i64, ptr %z, align 8
  %sub = sub i64 %0, %1
  ret i64 %sub
}

define i64 @g(i64 %a, i32 %b) #0 {
entry:
; V5-NOT: movw
; V6-NOT: movw
; V6T2: movw
; V7: movw
  %0 = mul i64 %a, 86400000
  %mul = add i64 %0, -210866803200000
  %conv = sext i32 %b to i64
  %add = add nsw i64 %mul, %conv
  ret i64 %add
}
