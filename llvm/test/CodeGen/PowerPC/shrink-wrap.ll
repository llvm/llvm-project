; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64le-unknown-unknown -mcpu=pwr9 | FileCheck  %s --check-prefixes=CHECK,CHECK64
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-ibm-aix-xcoff -mcpu=pwr9 -mattr=-altivec | FileCheck  %s --check-prefixes=CHECK,CHECK32
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-ibm-aix-xcoff -mcpu=pwr9 -mattr=-altivec | FileCheck  %s --check-prefixes=CHECK,CHECK64

define signext i32 @shrinkwrapme(i32 signext %a, i32 signext %lim) {
entry:
  %cmp5 = icmp sgt i32 %lim, 0
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

 for.body.preheader:                               ; preds = %entry
  br label %for.body

 for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

 for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  %Ret.0.lcssa = phi i32 [ 0, %entry ], [ %0, %for.cond.cleanup.loopexit ]
  ret i32 %Ret.0.lcssa

 for.body:                                         ; preds = %for.body.preheader, %for.body
  %i.07 = phi i32 [ %inc, %for.body ], [ 0, %for.body.preheader ]
  %Ret.06 = phi i32 [ %0, %for.body ], [ 0, %for.body.preheader ]
  %0 = tail call i32 asm "add $0, $1, $2", "=r,r,r,~{r14},~{r15},~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31}"(i32 %a, i32 %Ret.06)
  %inc = add nuw nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, %lim
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body

; CHECK-LABEL: {{[\.]?}}shrinkwrapme:
; CHECK:            # %bb.0:
; CHECK-NEXT:         cmpwi
; Prolog code
; CHECK64-COUNT-18:   std

; CHECK32-COUNT-18:   stw

; CHECK:              blt 0, {{.*}}BB0_3
; CHECK:            # %bb.1:
; CHECK:              li
; CHECK:            {{.*}}BB0_2:
; CHECK:              add
; CHECK:              bdnz {{.*}}BB0_2
; CHECK-NEXT:         b {{.*}}BB0_4
; CHECK:            {{.*}}BB0_3:
; CHECK-NEXT:         li
; CHECK:            {{.*}}BB0_4:

; Epilog code
; CHECK64-COUNT-18:   ld
;
; CHECK32-COUNT-18:   lwz

; CHECK:              blr
}
