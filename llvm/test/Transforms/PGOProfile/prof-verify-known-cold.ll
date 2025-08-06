; Test prof-verify for functions explicitly marked as cold

; RUN: opt -passes=prof-inject,prof-verify %s -o - 2>&1 | FileCheck %s

define void @foo(i32 %i) !prof !0 {
  %c = icmp eq i32 %i, 0
  br i1 %c, label %yes, label %no
yes:
  ret void
no:
  ret void
}
!0 = !{!"function_entry_count", i32 0}

; CHECK-NOT: Profile verification failed
