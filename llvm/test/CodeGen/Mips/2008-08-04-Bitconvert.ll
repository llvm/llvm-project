; RUN: llc -mtriple=mips < %s | FileCheck %s

define float @A(i32 %u) nounwind  {
entry:
; CHECK: mtc1 
  bitcast i32 %u to float
  ret float %0
}

define i32 @B(float %u) nounwind  {
entry:
; CHECK: mfc1 
  bitcast float %u to i32
  ret i32 %0
}
