; RUN: opt -S -passes=cross-dso-cfi < %s | FileCheck %s

; CHECK:     define void @__cfi_check(i64 %[[TYPE:.*]], ptr %[[ADDR:.*]], ptr %[[DATA:.*]]) align 4096
; CHECK:     switch i64 %[[TYPE]], label %[[FAIL:.*]] [
; CHECK-NEXT:   i64 111, label %[[L1:.*]]
; CHECK-NEXT:   i64 222, label %[[L2:.*]]
; CHECK-NEXT:   i64 333, label %[[L3:.*]]
; CHECK-NEXT:   i64 444, label %[[L4:.*]]
; CHECK-NEXT: {{]$}}

; CHECK:     [[EXIT:.*]]:
; CHECK-NEXT:   ret void

; CHECK:     [[FAIL]]:
; CHECK-NEXT:   call void @__cfi_check_fail(ptr %[[DATA]], ptr %[[ADDR]])
; CHECK-NEXT:   br label %[[EXIT]]

; CHECK:     [[L1]]:
; CHECK-NEXT:   call i1 @llvm.type.test(ptr %[[ADDR]], metadata i64 111)
; CHECK-NEXT:   br {{.*}} label %[[EXIT]], label %[[FAIL]]

; CHECK:     [[L2]]:
; CHECK-NEXT:   call i1 @llvm.type.test(ptr %[[ADDR]], metadata i64 222)
; CHECK-NEXT:   br {{.*}} label %[[EXIT]], label %[[FAIL]]

; CHECK:     [[L3]]:
; CHECK-NEXT:   call i1 @llvm.type.test(ptr %[[ADDR]], metadata i64 333)
; CHECK-NEXT:   br {{.*}} label %[[EXIT]], label %[[FAIL]]

; CHECK:     [[L4]]:
; CHECK-NEXT:   call i1 @llvm.type.test(ptr %[[ADDR]], metadata i64 444)
; CHECK-NEXT:   br {{.*}} label %[[EXIT]], label %[[FAIL]]

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZTV1A = constant i8 0, !type !4, !type !5
@_ZTV1B = constant i8 0, !type !4, !type !5, !type !6, !type !7

define signext i8 @f11() !type !0 !type !1 {
entry:
  ret i8 1
}

define signext i8 @f12() !type !0 !type !1 {
entry:
  ret i8 2
}

define signext i8 @f13() !type !0 !type !1 {
entry:
  ret i8 3
}

define i32 @f21() !type !2 !type !3 {
entry:
  ret i32 4
}

define i32 @f22() !type !2 !type !3 {
entry:
  ret i32 5
}

define weak_odr hidden void @__cfi_check_fail(ptr, ptr) {
entry:
  ret void
}

!llvm.module.flags = !{!8}

!0 = !{i64 0, !"_ZTSFcvE"}
!1 = !{i64 0, i64 111}
!2 = !{i64 0, !"_ZTSFivE"}
!3 = !{i64 0, i64 222}
!4 = !{i64 16, !"_ZTS1A"}
!5 = !{i64 16, i64 333}
!6 = !{i64 16, !"_ZTS1B"}
!7 = !{i64 16, i64 444}
!8 = !{i32 4, !"Cross-DSO CFI", i32 1}
