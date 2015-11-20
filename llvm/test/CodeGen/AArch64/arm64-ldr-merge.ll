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

; CHECK-LABEL: Ldrsh_merge
; CHECK: ldr [[NEW_DEST:w[0-9]+]]
; CHECK-DAG: asr [[LO_PART:w[0-9]+]], [[NEW_DEST]], #16
; CHECK-DAG: sxth [[HI_PART:w[0-9]+]], [[NEW_DEST]]
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]

define i32 @Ldrsh_merge(i16* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i16, i16* %p, i64 4
  %tmp = load i16, i16* %add.ptr0
  %add.ptr = getelementptr inbounds i16, i16* %p, i64 5
  %tmp1 = load i16, i16* %add.ptr
  %sexttmp = sext i16 %tmp to i32
  %sexttmp1 = sext i16 %tmp1 to i32
  %add = sub nsw i32 %sexttmp1, %sexttmp
  ret i32 %add
}

; CHECK-LABEL: Ldrsh_zsext_merge
; CHECK: ldr [[NEW_DEST:w[0-9]+]]
; LE-DAG: and [[LO_PART:w[0-9]+]], [[NEW_DEST]], #0xffff
; LE-DAG: asr [[HI_PART:w[0-9]+]], [[NEW_DEST]], #16
; BE-DAG: sxth [[LO_PART:w[0-9]+]], [[NEW_DEST]]
; BE-DAG: lsr [[HI_PART:w[0-9]+]], [[NEW_DEST]], #16
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldrsh_zsext_merge(i16* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i16, i16* %p, i64 4
  %tmp = load i16, i16* %add.ptr0
  %add.ptr = getelementptr inbounds i16, i16* %p, i64 5
  %tmp1 = load i16, i16* %add.ptr
  %sexttmp = zext i16 %tmp to i32
  %sexttmp1 = sext i16 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldrsh_szext_merge
; CHECK: ldr [[NEW_DEST:w[0-9]+]]
; LE-DAG: sxth [[LO_PART:w[0-9]+]], [[NEW_DEST]]
; LE-DAG: lsr [[HI_PART:w[0-9]+]], [[NEW_DEST]], #16
; BE-DAG: and [[LO_PART:w[0-9]+]], [[NEW_DEST]], #0xffff
; BE-DAG: asr [[HI_PART:w[0-9]+]], [[NEW_DEST]], #16
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldrsh_szext_merge(i16* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i16, i16* %p, i64 4
  %tmp = load i16, i16* %add.ptr0
  %add.ptr = getelementptr inbounds i16, i16* %p, i64 5
  %tmp1 = load i16, i16* %add.ptr
  %sexttmp = sext i16 %tmp to i32
  %sexttmp1 = zext i16 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldrb_merge
; CHECK: ldrh [[NEW_DEST:w[0-9]+]]
; CHECK-DAG: and [[LO_PART:w[0-9]+]], [[NEW_DEST]], #0xff
; CHECK-DAG: ubfx [[HI_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldrb_merge(i8* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i8, i8* %p, i64 2
  %tmp = load i8, i8* %add.ptr0
  %add.ptr = getelementptr inbounds i8, i8* %p, i64 3
  %tmp1 = load i8, i8* %add.ptr
  %sexttmp = zext i8 %tmp to i32
  %sexttmp1 = zext i8 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldrsb_merge
; CHECK: ldrh [[NEW_DEST:w[0-9]+]]
; CHECK-DAG: sxtb [[LO_PART:w[0-9]+]], [[NEW_DEST]]
; CHECK-DAG: sbfx [[HI_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldrsb_merge(i8* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i8, i8* %p, i64 2
  %tmp = load i8, i8* %add.ptr0
  %add.ptr = getelementptr inbounds i8, i8* %p, i64 3
  %tmp1 = load i8, i8* %add.ptr
  %sexttmp = sext i8 %tmp to i32
  %sexttmp1 = sext i8 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldrsb_zsext_merge
; CHECK: ldrh [[NEW_DEST:w[0-9]+]]
; LE-DAG: and [[LO_PART:w[0-9]+]], [[NEW_DEST]], #0xff
; LE-DAG: sbfx [[HI_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; BE-DAG: sxtb [[LO_PART:w[0-9]+]], [[NEW_DEST]]
; BE-DAG: ubfx [[HI_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldrsb_zsext_merge(i8* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i8, i8* %p, i64 2
  %tmp = load i8, i8* %add.ptr0
  %add.ptr = getelementptr inbounds i8, i8* %p, i64 3
  %tmp1 = load i8, i8* %add.ptr
  %sexttmp = zext i8 %tmp to i32
  %sexttmp1 = sext i8 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldrsb_szext_merge
; CHECK: ldrh [[NEW_DEST:w[0-9]+]]
; LE-DAG: sxtb [[LO_PART:w[0-9]+]], [[NEW_DEST]]
; LE-DAG: ubfx [[HI_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; BE-DAG: and [[LO_PART:w[0-9]+]], [[NEW_DEST]], #0xff
; BE-DAG: sbfx [[HI_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldrsb_szext_merge(i8* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i8, i8* %p, i64 2
  %tmp = load i8, i8* %add.ptr0
  %add.ptr = getelementptr inbounds i8, i8* %p, i64 3
  %tmp1 = load i8, i8* %add.ptr
  %sexttmp = sext i8 %tmp to i32
  %sexttmp1 = zext i8 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldursh_merge
; CHECK: ldur [[NEW_DEST:w[0-9]+]]
; CHECK-DAG: asr  [[LO_PART:w[0-9]+]], [[NEW_DEST]], #16
; CHECK-DAG: sxth [[HI_PART:w[0-9]+]], [[NEW_DEST]]
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldursh_merge(i16* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i16, i16* %p, i64 -1
  %tmp = load i16, i16* %add.ptr0
  %add.ptr = getelementptr inbounds i16, i16* %p, i64 -2
  %tmp1 = load i16, i16* %add.ptr
  %sexttmp = sext i16 %tmp to i32
  %sexttmp1 = sext i16 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldursh_zsext_merge
; CHECK: ldur [[NEW_DEST:w[0-9]+]]
; LE-DAG: lsr  [[LO_PART:w[0-9]+]], [[NEW_DEST]], #16
; LE-DAG: sxth [[HI_PART:w[0-9]+]], [[NEW_DEST]]
; BE-DAG: asr  [[LO_PART:w[0-9]+]], [[NEW_DEST]], #16
; BE-DAG: and [[HI_PART:w[0-9]+]], [[NEW_DEST]], #0xffff
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldursh_zsext_merge(i16* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i16, i16* %p, i64 -1
  %tmp = load i16, i16* %add.ptr0
  %add.ptr = getelementptr inbounds i16, i16* %p, i64 -2
  %tmp1 = load i16, i16* %add.ptr
  %sexttmp = zext i16 %tmp to i32
  %sexttmp1 = sext i16 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldursh_szext_merge
; CHECK: ldur [[NEW_DEST:w[0-9]+]]
; LE-DAG: asr  [[LO_PART:w[0-9]+]], [[NEW_DEST]], #16
; LE-DAG: and [[HI_PART:w[0-9]+]], [[NEW_DEST]], #0xffff
; BE-DAG: lsr  [[LO_PART:w[0-9]+]], [[NEW_DEST]], #16
; BE-DAG: sxth [[HI_PART:w[0-9]+]], [[NEW_DEST]]
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldursh_szext_merge(i16* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i16, i16* %p, i64 -1
  %tmp = load i16, i16* %add.ptr0
  %add.ptr = getelementptr inbounds i16, i16* %p, i64 -2
  %tmp1 = load i16, i16* %add.ptr
  %sexttmp = sext i16 %tmp to i32
  %sexttmp1 = zext i16 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldurb_merge
; CHECK: ldurh [[NEW_DEST:w[0-9]+]]
; CHECK-DAG: ubfx  [[LO_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; CHECK-DAG: and [[HI_PART:w[0-9]+]], [[NEW_DEST]], #0xff
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldurb_merge(i8* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i8, i8* %p, i64 -1
  %tmp = load i8, i8* %add.ptr0
  %add.ptr = getelementptr inbounds i8, i8* %p, i64 -2
  %tmp1 = load i8, i8* %add.ptr
  %sexttmp = zext i8 %tmp to i32
  %sexttmp1 = zext i8 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldursb_merge
; CHECK: ldurh [[NEW_DEST:w[0-9]+]]
; CHECK-DAG: sbfx [[LO_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; CHECK-DAG: sxtb [[HI_PART:w[0-9]+]], [[NEW_DEST]]
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldursb_merge(i8* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i8, i8* %p, i64 -1
  %tmp = load i8, i8* %add.ptr0
  %add.ptr = getelementptr inbounds i8, i8* %p, i64 -2
  %tmp1 = load i8, i8* %add.ptr
  %sexttmp = sext i8 %tmp to i32
  %sexttmp1 = sext i8 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldursb_zsext_merge
; CHECK: ldurh [[NEW_DEST:w[0-9]+]]
; LE-DAG: ubfx [[LO_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; LE-DAG: sxtb [[HI_PART:w[0-9]+]], [[NEW_DEST]]
; BE-DAG: sbfx [[LO_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; BE-DAG: and [[HI_PART:w[0-9]+]], [[NEW_DEST]], #0xff
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldursb_zsext_merge(i8* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i8, i8* %p, i64 -1
  %tmp = load i8, i8* %add.ptr0
  %add.ptr = getelementptr inbounds i8, i8* %p, i64 -2
  %tmp1 = load i8, i8* %add.ptr
  %sexttmp = zext i8 %tmp to i32
  %sexttmp1 = sext i8 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

; CHECK-LABEL: Ldursb_szext_merge
; CHECK: ldurh [[NEW_DEST:w[0-9]+]]
; LE-DAG: sbfx [[LO_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; LE-DAG: and [[HI_PART:w[0-9]+]], [[NEW_DEST]], #0xff
; BE-DAG: ubfx [[LO_PART:w[0-9]+]], [[NEW_DEST]], #8, #8
; BE-DAG: sxtb [[HI_PART:w[0-9]+]], [[NEW_DEST]]
; LE: sub {{w[0-9]+}}, [[LO_PART]], [[HI_PART]]
; BE: sub {{w[0-9]+}}, [[HI_PART]], [[LO_PART]]
define i32 @Ldursb_szext_merge(i8* %p) nounwind {
  %add.ptr0 = getelementptr inbounds i8, i8* %p, i64 -1
  %tmp = load i8, i8* %add.ptr0
  %add.ptr = getelementptr inbounds i8, i8* %p, i64 -2
  %tmp1 = load i8, i8* %add.ptr
  %sexttmp = sext i8 %tmp to i32
  %sexttmp1 = zext i8 %tmp1 to i32
  %add = sub nsw i32 %sexttmp, %sexttmp1
  ret i32 %add
}

