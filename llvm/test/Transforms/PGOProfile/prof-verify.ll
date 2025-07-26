; Test prof-inject and prof-verify

; RUN: opt -passes=prof-inject %s -S -o - | FileCheck %s --check-prefix=INJECT
; RUN: not opt -passes=prof-verify %s -S -o - 2>&1 | FileCheck %s --check-prefix=VERIFY
; RUN: opt -passes=prof-inject,prof-verify %s --disable-output
; RUN: opt -enable-profcheck %s -S -o - | FileCheck %s --check-prefix=INJECT

define void @foo(i32 %i) {
  %c = icmp eq i32 %i, 0
  br i1 %c, label %yes, label %no
yes:
  ret void
no:
  ret void
}

; INJECT: br i1 %c, label %yes, label %no, !prof !0
; INJECT: !0 = !{!"branch_weights", i32 3, i32 5}

; VERIFY: Profile verification failed