; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s
;
; Check that DAGCombiner does not crash after producing an illegal
; BUILD_VECTOR node.


define void @pr32422() {
; CHECK:        cdbr    %f0, %f0
; CHECK:        jo      .LBB0_1

BB:
  %I = insertelement <8 x i8> zeroinitializer, i8 -95, i32 3
  %I8 = insertelement <8 x i8> zeroinitializer, i8 -119, i32 2
  %FC = uitofp <8 x i8> %I8 to <8 x float>
  %Cmp18 = fcmp uno <8 x float> zeroinitializer, %FC
  %I22 = insertelement <8 x i1> %Cmp18, i1 true, i32 5
  br label %CF

CF:                                               ; preds = %CF, %BB
  %Cmp40 = fcmp uno double 0xC663C682E9619F00, undef
  br i1 %Cmp40, label %CF, label %CF353

CF353:                                            ; preds = %CF
  %E195 = extractelement <8 x i1> %I22, i32 4
  ret void
}
