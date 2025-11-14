; This test fails under the profcheck configuration due to profcheck creating
; metadata.
; UNSUPPORTED: profcheck

; RUN: split-file %s %t

; RUN: opt -passes=prof-inject %t/inject.ll -S -o - | FileCheck %t/inject.ll

; RUN: opt -passes=prof-inject %t/inject-some.ll \
; RUN:   -profcheck-default-select-true-weight=1 -profcheck-default-select-false-weight=6 \
; RUN:   -S -o - | FileCheck %t/inject-some.ll

; RUN: opt -passes=prof-verify %t/verify.ll 2>&1 | FileCheck %t/verify.ll

; RUN: not opt -passes=prof-verify %t/verify-missing.ll 2>&1 | FileCheck %t/verify-missing.ll

; verify we can disable it. It's sufficient to see opt not failing. 
; RUN: opt -passes=prof-verify -profcheck-annotate-select=0 --disable-output %t/verify-missing.ll

; verify vector selects without profiles are OK. It's sufficient opt doesn't fail.
; RUN: opt -passes=prof-verify --disable-output %t/verify-vec.ll


;--- inject.ll
declare void @foo(i32 %a);
define void @bar(i1 %c) {
  %v = select i1 %c, i32 1, i32 2
  call void @foo(i32 %v)
  ret void
}

define <2 x i32> @vec(<2 x i1> %c, <2 x i32> %v1, <2 x i32> %v2) {
  %r = select <2 x i1> %c, <2 x i32> %v1, <2 x i32> %v2
  ret <2 x i32> %r
}

; CHECK-LABEL: @bar
; CHECK: %v = select i1 %c, i32 1, i32 2, !prof !1
; CHECK-LABEL: @vec
; CHECK-NOT: select {{.*}} !prof
; CHECK: !0 = !{!"function_entry_count", i64 1000}
; CHECK: !1 = !{!"branch_weights", i32 2, i32 3}

;--- inject-some.ll
declare void @foo(i32 %a);
define void @bar(i1 %c) {
  %e = select i1 %c, i32 1, i32 2, !prof !0
  %c2 = icmp eq i32 %e, 2
  %v = select i1 %c2, i32 5, i32 10
  call void @foo(i32 %v)
  ret void
}
!0 = !{!"branch_weights", i32 2, i32 3}
; CHECK-LABEL: @bar
; CHECK: %v = select i1 %c2, i32 5, i32 10, !prof !2
; CHECK: !0 = !{!"function_entry_count", i64 1000}
; CHECK: !1 = !{!"branch_weights", i32 2, i32 3}
; CHECK: !2 = !{!"branch_weights", i32 1, i32 6}

;--- verify.ll
declare void @foo(i32 %a);
define void @bar(i1 %c) !prof !0 {
  %v = select i1 %c, i32 1, i32 2, !prof !1
  call void @foo(i32 %v)
  ret void
}
!0 = !{!"function_entry_count", i64 1000}
!1 = !{!"branch_weights", i32 1, i32 7}
; CHECK-NOT: Profile verification failed: select annotation missing

;--- verify-missing.ll
declare void @foo(i32 %a);
define void @bar(i1 %c) !prof !0 {
  %v = select i1 %c, i32 1, i32 2
  call void @foo(i32 %v)
  ret void
}
!0 = !{!"function_entry_count", i64 1000}
; CHECK: Profile verification failed: select annotation missing

;--- verify-vec.ll
define <2 x i32> @vec(<2 x i1> %c, <2 x i32> %v1, <2 x i32> %v2) !prof !{!"function_entry_count", i32 10} {
  %r = select <2 x i1> %c, <2 x i32> %v1, <2 x i32> %v2
  ret <2 x i32> %r
}
