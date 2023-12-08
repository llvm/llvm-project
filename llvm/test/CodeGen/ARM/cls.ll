; RUN: llc -mtriple=armv5 %s -o - | FileCheck %s

; CHECK:      eor [[T:r[0-9]+]], [[T]], [[T]], asr #31
; CHECK-NEXT: mov [[C1:r[0-9]+]], #1
; CHECK-NEXT: orr [[T]], [[C1]], [[T]], lsl #1
; CHECK-NEXT: clz [[T]], [[T]]
define i32 @cls(i32 %t) {
  %cls.i = call i32 @llvm.arm.cls(i32 %t)
  ret i32 %cls.i
}

; CHECK: cmp r1, #0
; CHECK: mvnne [[ADJUSTEDLO:r[0-9]+]], r0
; CHECK: clz [[CLZLO:r[0-9]+]], [[ADJUSTEDLO]]
; CHECK: eor [[A:r[0-9]+]], r1, r1, asr #31
; CHECK: mov r1, #1
; CHECK: orr [[A]], r1, [[A]], lsl #1
; CHECK: clz [[CLSHI:r[0-9]+]], [[A]]
; CHECK: cmp [[CLSHI]], #31
; CHECK: addeq r0, [[CLZLO]], #31
define i32 @cls64(i64 %t) {
  %cls.i = call i32 @llvm.arm.cls64(i64 %t)
  ret i32 %cls.i
}

declare i32 @llvm.arm.cls(i32) nounwind
declare i32 @llvm.arm.cls64(i64) nounwind
