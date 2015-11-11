; RUN: llc < %s -mtriple aarch64--none-eabi -mcpu=cortex-a57 -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=LE
; RUN: llc < %s -mtriple aarch64_be--none-eabi -mcpu=cortex-a57 -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=BE

; CHECK-LABEL: Ldrh_merge
; CHECK-NOT: ldrh
; CHECK: ldr [[NEW_DEST:w[0-9]+]]
; CHECK-DAG: and [[LO_PART:w[0-9]+]], [[NEW_DEST]], #0xffff
; CHECK-DAG: lsr [[HI_PART:w[0-9]+]], [[NEW_DEST]], #16
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i16 @Ldrh_merge(i16* nocapture readonly %p) {
  %1 = load i16, i16* %p, align 2
  %arrayidx2 = getelementptr inbounds i16, i16* %p, i64 1
  %2 = load i16, i16* %arrayidx2, align 2
  %add = sub nuw nsw i16 %1, %2
  ret i16 %add
}

; CHECK-LABEL: Ldurh_merge
; CHECK-NOT: ldurh
; CHECK: ldur [[NEW_DEST:w[0-9]+]]
; CHECK-DAG: and [[LO_PART:w[0-9]+]], [[NEW_DEST]], #0xffff
; CHECK-DAG: lsr  [[HI_PART:w[0-9]+]], [[NEW_DEST]]
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i16 @Ldurh_merge(i16* nocapture readonly %p)  {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %p, i64 -2
  %0 = load i16, i16* %arrayidx
  %arrayidx3 = getelementptr inbounds i16, i16* %p, i64 -1
  %1 = load i16, i16* %arrayidx3
  %add = sub nuw nsw i16 %0, %1
  ret i16 %add
}

; CHECK-LABEL: Ldrh_4_merge
; CHECK-NOT: ldrh
; CHECK: ldp [[WORD1:w[0-9]+]], [[WORD2:w[0-9]+]], [x0]
; CHECK-DAG: and [[WORD1LO:w[0-9]+]], [[WORD1]], #0xffff
; CHECK-DAG: lsr [[WORD1HI:w[0-9]+]], [[WORD1]], #16
; CHECK-DAG: and [[WORD2LO:w[0-9]+]], [[WORD2]], #0xffff
; CHECK-DAG: lsr [[WORD2HI:w[0-9]+]], [[WORD2]], #16
; LE-DAG: sub [[TEMP1:w[0-9]+]], [[WORD1HI]], [[WORD1LO]]
; BE-DAG: sub [[TEMP1:w[0-9]+]], [[WORD1LO]], [[WORD1HI]]
; LE: udiv [[TEMP2:w[0-9]+]], [[TEMP1]], [[WORD2LO]]
; BE: udiv [[TEMP2:w[0-9]+]], [[TEMP1]], [[WORD2HI]]
; LE: sub w0, [[TEMP2]], [[WORD2HI]]
; BE: sub w0, [[TEMP2]], [[WORD2LO]]
define i16 @Ldrh_4_merge(i16* nocapture readonly %P) {
  %arrayidx = getelementptr inbounds i16, i16* %P, i64 0
  %l0 = load i16, i16* %arrayidx
  %arrayidx2 = getelementptr inbounds i16, i16* %P, i64 1
  %l1 = load i16, i16* %arrayidx2
  %arrayidx7 = getelementptr inbounds i16, i16* %P, i64 2
  %l2 = load i16, i16* %arrayidx7
  %arrayidx12 = getelementptr inbounds i16, i16* %P, i64 3
  %l3 = load i16, i16* %arrayidx12
  %add4 = sub nuw nsw i16 %l1, %l0
  %add9 = udiv i16 %add4, %l2
  %add14 = sub nuw nsw i16 %add9, %l3
  ret i16 %add14
}
