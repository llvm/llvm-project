; RUN: opt -S -cross-dso-cfi < %s | FileCheck %s

; CHECK:     define void @__cfi_check(i64 %[[TYPE:.*]], i8* %[[ADDR:.*]], i8* %[[DATA:.*]]) align 4096
; CHECK:     switch i64 %[[TYPE]], label %[[FAIL:.*]] [
; CHECK-NEXT:   i64 111, label %[[L1:.*]]
; CHECK-NEXT:   i64 222, label %[[L2:.*]]
; CHECK-NEXT:   i64 333, label %[[L3:.*]]
; CHECK-NEXT:   i64 444, label %[[L4:.*]]
; CHECK-NEXT: {{]$}}

; CHECK:     [[EXIT:.*]]:
; CHECK-NEXT:   ret void

; CHECK:     [[FAIL]]:
; CHECK-NEXT:   call void @__cfi_check_fail(i8* %[[DATA]], i8* %[[ADDR]])
; CHECK-NEXT:   br label %[[EXIT]]

; CHECK:     [[L1]]:
; CHECK-NEXT:   call i1 @llvm.bitset.test(i8* %[[ADDR]], metadata i64 111)
; CHECK-NEXT:   br {{.*}} label %[[EXIT]], label %[[FAIL]]

; CHECK:     [[L2]]:
; CHECK-NEXT:   call i1 @llvm.bitset.test(i8* %[[ADDR]], metadata i64 222)
; CHECK-NEXT:   br {{.*}} label %[[EXIT]], label %[[FAIL]]

; CHECK:     [[L3]]:
; CHECK-NEXT:   call i1 @llvm.bitset.test(i8* %[[ADDR]], metadata i64 333)
; CHECK-NEXT:   br {{.*}} label %[[EXIT]], label %[[FAIL]]

; CHECK:     [[L4]]:
; CHECK-NEXT:   call i1 @llvm.bitset.test(i8* %[[ADDR]], metadata i64 444)
; CHECK-NEXT:   br {{.*}} label %[[EXIT]], label %[[FAIL]]

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV1A = constant i8 0
@_ZTI1A = constant i8 0
@_ZTS1A = constant i8 0
@_ZTV1B = constant i8 0
@_ZTI1B = constant i8 0
@_ZTS1B = constant i8 0

define signext i8 @f11() {
entry:
  ret i8 1
}

define signext i8 @f12() {
entry:
  ret i8 2
}

define signext i8 @f13() {
entry:
  ret i8 3
}

define i32 @f21() {
entry:
  ret i32 4
}

define i32 @f22() {
entry:
  ret i32 5
}

define weak_odr hidden void @__cfi_check_fail(i8*, i8*) {
entry:
  ret void
}

!llvm.bitsets = !{!0, !1, !2, !3, !4, !7, !8, !9, !10, !11, !12, !13, !14, !15}
!llvm.module.flags = !{!17}

!0 = !{!"_ZTSFcvE", i8 ()* @f11, i64 0}
!1 = !{i64 111, i8 ()* @f11, i64 0}
!2 = !{!"_ZTSFcvE", i8 ()* @f12, i64 0}
!3 = !{i64 111, i8 ()* @f12, i64 0}
!4 = !{!"_ZTSFcvE", i8 ()* @f13, i64 0}
!5 = !{i64 111, i8 ()* @f13, i64 0}
!6 = !{!"_ZTSFivE", i32 ()* @f21, i64 0}
!7 = !{i64 222, i32 ()* @f21, i64 0}
!8 = !{!"_ZTSFivE", i32 ()* @f22, i64 0}
!9 = !{i64 222, i32 ()* @f22, i64 0}
!10 = !{!"_ZTS1A", i8* @_ZTV1A, i64 16}
!11 = !{i64 333, i8* @_ZTV1A, i64 16}
!12 = !{!"_ZTS1A", i8* @_ZTV1B, i64 16}
!13 = !{i64 333, i8* @_ZTV1B, i64 16}
!14 = !{!"_ZTS1B", i8* @_ZTV1B, i64 16}
!15 = !{i64 444, i8* @_ZTV1B, i64 16}
!17= !{i32 4, !"Cross-DSO CFI", i32 1}
