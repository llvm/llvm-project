; RUN: llc -verify-machineinstrs -mcpu=a2 < %s | FileCheck %s -check-prefix=INVFUNCDESC
; RUN: llc -verify-machineinstrs -mcpu=a2 -mattr=-invariant-function-descriptors < %s | FileCheck %s -check-prefix=NONINVFUNCDESC
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define void @bar(void (...)* nocapture %x) #0 {
entry:
  %callee.knr.cast = bitcast void (...)* %x to void ()*
  br label %for.body

; INVFUNCDESC-LABEL: @bar
; INVFUNCDESC-DAG: ld [[REG1:[0-9]+]], 8(3)
; INVFUNCDESC-DAG: ld [[REG2:[0-9]+]], 16(3)
; INVFUNCDESC-DAG: ld [[REG3:[0-9]+]], 0(3)

; INVFUNCDESC: %for.body
; INVFUNCDESC-DAG: mtctr [[REG3]]
; INVFUNCDESC-DAG: mr 11, [[REG2]]
; INVFUNCDESC-DAG: mr 2, [[REG1]]
; INVFUNCDESC: bctrl
; INVFUNCDESC-NEXT: ld 2, 40(1)

; NONINVFUNCDESC-LABEL: @bar
; NONINVFUNCDESC: %for.body
; NONINVFUNCDESC-DAG: ld 3, 0(30)
; NONINVFUNCDESC-DAG: ld 11, 16(30)
; NONINVFUNCDESC-DAG: ld 2, 8(30)
; NONINVFUNCDESC: mtctr 3
; NONINVFUNCDESC: bctrl
; NONINVFUNCDESC-NEXT: ld 2, 40(1)

for.body:                                         ; preds = %for.body, %entry
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  tail call void %callee.knr.cast() #0
  %inc = add nuw nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, 1600000000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { nounwind }

