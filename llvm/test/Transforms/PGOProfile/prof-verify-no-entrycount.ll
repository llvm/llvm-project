; This test fails under the profcheck configuration due to profcheck creating
; metadata.
; UNSUPPORTED: profcheck

; Test prof-verify for functions without entry count

; RUN: not opt -passes=prof-verify %s -o - 2>&1 | FileCheck %s

define void @foo(i32 %i) {
  %c = icmp eq i32 %i, 0
  br i1 %c, label %yes, label %no
yes:
  ret void
no:
  ret void
}

; CHECK: Profile verification failed: function entry count missing (set to 0 if cold)
