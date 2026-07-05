; RUN: llc -mtriple=s390x-linux-gnu -mcpu=z16 < %s  | FileCheck %s
;
; Check that DAGCombiner doesn't crash in SystemZ combineExtract()
; when handling EXTRACT_VECTOR_ELT with a vector of i1:s.

define i32 @fun(i32 %arg) {
; CHECK-LABEL: fun:
entry:
  %cc = icmp eq i32 %arg, 0
  br label %loop

loop:
  %P = phi <128 x i1> [ zeroinitializer, %entry ], [ bitcast (<2 x i64> <i64 3, i64 3> to <128 x i1>), %loop ]
  br i1 %cc, label %exit, label %loop

exit:
  %E = extractelement <128 x i1> %P, i64 0
  %Res = zext i1 %E to i32
  ret i32 %Res
}
