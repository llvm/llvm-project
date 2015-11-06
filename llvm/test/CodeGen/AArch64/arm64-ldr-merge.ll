; RUN: llc < %s -march=arm64 -mcpu=cortex-a57 -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: Ldrh_merge
; CHECK-NOT: ldrh
; CHECK: ldr [[NEW_DEST:w[0-9]+]]
; CHECK: and w{{[0-9]+}}, [[NEW_DEST]], #0xffff
; CHECK: lsr  w{{[0-9]+}}, [[NEW_DEST]]
define i16 @Ldrh_merge(i16* nocapture readonly %p) {
  %1 = load i16, i16* %p, align 2
  %arrayidx2 = getelementptr inbounds i16, i16* %p, i64 1
  %2 = load i16, i16* %arrayidx2, align 2
  %add = add nuw nsw i16 %1, %2
  ret i16 %add
}

; CHECK-LABEL: Ldurh_merge
; CHECK-NOT: ldurh
; CHECK: ldur [[NEW_DEST:w[0-9]+]]
; CHECK: and w{{[0-9]+}}, [[NEW_DEST]], #0xffff
; CHECK: lsr  w{{[0-9]+}}, [[NEW_DEST]]
define i16 @Ldurh_merge(i16* nocapture readonly %p)  {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %p, i64 -2
  %0 = load i16, i16* %arrayidx
  %arrayidx3 = getelementptr inbounds i16, i16* %p, i64 -1
  %1 = load i16, i16* %arrayidx3
  %add = add nuw nsw i16 %0, %1
  ret i16 %add
}

; CHECK-LABEL: Ldrh_4_merge
; CHECK-NOT: ldrh
; CHECK: ldp [[NEW_DEST:w[0-9]+]]
define i16 @Ldrh_4_merge(i16* nocapture readonly %P) {
  %arrayidx = getelementptr inbounds i16, i16* %P, i64 0
  %l0 = load i16, i16* %arrayidx
  %arrayidx2 = getelementptr inbounds i16, i16* %P, i64 1
  %l1 = load i16, i16* %arrayidx2
  %arrayidx7 = getelementptr inbounds i16, i16* %P, i64 2
  %l2 = load i16, i16* %arrayidx7
  %arrayidx12 = getelementptr inbounds i16, i16* %P, i64 3
  %l3 = load i16, i16* %arrayidx12
  %add4 = add nuw nsw i16 %l1, %l0
  %add9 = add nuw nsw i16 %add4, %l2
  %add14 = add nuw nsw i16 %add9, %l3
  ret i16 %add14
}
