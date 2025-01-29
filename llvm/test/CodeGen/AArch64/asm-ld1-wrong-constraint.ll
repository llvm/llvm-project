; RUN: not llc < %s -o - 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx"

%struct.uint64x2x4_t = type { [4 x <2 x i64>] }

define i64 @rdar130887714(ptr noundef %0) {
  %2 = alloca ptr, align 8
  %3 = alloca %struct.uint64x2x4_t, align 16
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  call void asm sideeffect "ld1 { $0.2d, ${0:T}.2d, ${0:U}.2d, ${0:V}.2d} , [$1]", "*w,r"(ptr elementtype(%struct.uint64x2x4_t) %3, ptr %4) #0, !srcloc !0
; CHECK: error: Don't know how to handle indirect register inputs yet for constraint 'w' at line 250

  %5 = getelementptr inbounds %struct.uint64x2x4_t, ptr %3, i32 0, i32 0
  %6 = getelementptr inbounds [4 x <2 x i64>], ptr %5, i64 0, i64 0
  %7 = load <2 x i64>, ptr %6, align 16
  %8 = extractelement <2 x i64> %7, i32 1
  ret i64 %8
}

!0 = !{i64 250}