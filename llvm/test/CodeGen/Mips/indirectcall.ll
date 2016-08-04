; RUN: llc  < %s -march=mipsel -relocation-model=static | FileCheck %s 

define void @foo0(void (i32)* nocapture %f1) nounwind {
entry:
; CHECK: jr $25
  tail call void %f1(i32 13) nounwind
  ret void
}
