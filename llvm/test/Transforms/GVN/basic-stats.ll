; REQUIRES: asserts
; RUN: opt < %s -stats -passes=gvn -o /dev/null 2>&1 | FileCheck %s
define i32 @main() {
block1:
  %z1 = bitcast i32 0 to i32
  br label %block2
block2:
  %z2 = bitcast i32 0 to i32
  ret i32 %z2
}

; CHECK: 1 gvn - Number of blocks merged
; CHECK: 2 gvn - Number of instructions deleted
; CHECK: 2 gvn - Number of instructions simplified