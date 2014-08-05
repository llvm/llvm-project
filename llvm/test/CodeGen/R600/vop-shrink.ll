; RUN: llc -march=r600 -mcpu=SI -verify-machineinstrs< %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; XXX: This testis for a bug in the SIShrinkInstruction pass and it will be
;       relevant once we are selecting 64-bit instructions.  We are
;       currently selecting mostly 32-bit instruction, so the
;       SIShrinkInstructions pass isn't doing much.
; XFAIL: *

; Test that we correctly commute a sub instruction
; FUNC-LABEL: @sub_rev
; SI-NOT: V_SUB_I32_e32 v{{[0-9]+}}, s
; SI: V_SUBREV_I32_e32 v{{[0-9]+}}, s

; ModuleID = 'vop-shrink.ll'

define void @sub_rev(i32 addrspace(1)* %out, <4 x i32> %sgpr, i32 %cond) {
entry:
  %vgpr = call i32 @llvm.r600.read.tidig.x() #1
  %tmp = icmp eq i32 %cond, 0
  br i1 %tmp, label %if, label %else

if:                                               ; preds = %entry
  %tmp1 = getelementptr i32 addrspace(1)* %out, i32 1
  %tmp2 = extractelement <4 x i32> %sgpr, i32 1
  store i32 %tmp2, i32 addrspace(1)* %out
  br label %endif

else:                                             ; preds = %entry
  %tmp3 = extractelement <4 x i32> %sgpr, i32 2
  %tmp4 = sub i32 %vgpr, %tmp3
  store i32 %tmp4, i32 addrspace(1)* %out
  br label %endif

endif:                                            ; preds = %else, %if
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.x() #0

attributes #0 = { nounwind readnone }
attributes #1 = { readnone }
