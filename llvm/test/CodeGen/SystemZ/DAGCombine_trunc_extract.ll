; RUN: llc -mtriple=s390x-linux-gnu -mcpu=zEC12 < %s  | FileCheck %s
;
; Check that DAGCombiner doesn't crash in SystemZ combineTruncateExtract()
; when handling EXTRACT_VECTOR_ELT without vector support.

define void @autogen_SD21598(<2 x i8> %Arg) {
; CHECK:	stc	%r3, 0(%r1)
; CHECK:	j	.LBB0_1

entry:
  br label %loop

loop:                                            ; preds = %CF249, %CF247
  %Shuff = shufflevector <2 x i8> undef, <2 x i8> %Arg, <2 x i32> <i32 3, i32 1>
  %E = extractelement <2 x i8> %Shuff, i32 0
  store i8 %E, i8* undef
  br label %loop
}
