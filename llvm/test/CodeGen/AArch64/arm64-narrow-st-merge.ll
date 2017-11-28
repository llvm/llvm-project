; RUN: llc < %s -mtriple aarch64--none-eabi -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple aarch64--none-eabi -mattr=+strict-align -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-STRICT

; CHECK-LABEL: Strh_zero
; CHECK: str wzr
; CHECK-STRICT-LABEL: Strh_zero
; CHECK-STRICT: strh wzr
; CHECK-STRICT: strh wzr
define void @Strh_zero(i16* nocapture %P, i32 %n) {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i16, i16* %P, i64 %idxprom
  store i16 0, i16* %arrayidx
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i16, i16* %P, i64 %idxprom1
  store i16 0, i16* %arrayidx2
  ret void
}

; CHECK-LABEL: Strh_zero_4
; CHECK: str xzr
; CHECK-STRICT-LABEL: Strh_zero_4
; CHECK-STRICT: strh wzr
; CHECK-STRICT: strh wzr
; CHECK-STRICT: strh wzr
; CHECK-STRICT: strh wzr
define void @Strh_zero_4(i16* nocapture %P, i32 %n) {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i16, i16* %P, i64 %idxprom
  store i16 0, i16* %arrayidx
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i16, i16* %P, i64 %idxprom1
  store i16 0, i16* %arrayidx2
  %add3 = add nsw i32 %n, 2
  %idxprom4 = sext i32 %add3 to i64
  %arrayidx5 = getelementptr inbounds i16, i16* %P, i64 %idxprom4
  store i16 0, i16* %arrayidx5
  %add6 = add nsw i32 %n, 3
  %idxprom7 = sext i32 %add6 to i64
  %arrayidx8 = getelementptr inbounds i16, i16* %P, i64 %idxprom7
  store i16 0, i16* %arrayidx8
  ret void
}

; CHECK-LABEL: Strw_zero
; CHECK: str xzr
; CHECK-STRICT-LABEL: Strw_zero
; CHECK-STRICT: stp wzr, wzr
define void @Strw_zero(i32* nocapture %P, i32 %n) {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i32, i32* %P, i64 %idxprom
  store i32 0, i32* %arrayidx
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %P, i64 %idxprom1
  store i32 0, i32* %arrayidx2
  ret void
}

; CHECK-LABEL: Strw_zero_nonzero
; CHECK: stp wzr, w1
define void @Strw_zero_nonzero(i32* nocapture %P, i32 %n)  {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i32, i32* %P, i64 %idxprom
  store i32 0, i32* %arrayidx
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %P, i64 %idxprom1
  store i32 %n, i32* %arrayidx2
  ret void
}

; CHECK-LABEL: Strw_zero_4
; CHECK: stp xzr, xzr
; CHECK-STRICT-LABEL: Strw_zero_4
; CHECK-STRICT: stp wzr, wzr
; CHECK-STRICT: stp wzr, wzr
define void @Strw_zero_4(i32* nocapture %P, i32 %n) {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i32, i32* %P, i64 %idxprom
  store i32 0, i32* %arrayidx
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %P, i64 %idxprom1
  store i32 0, i32* %arrayidx2
  %add3 = add nsw i32 %n, 2
  %idxprom4 = sext i32 %add3 to i64
  %arrayidx5 = getelementptr inbounds i32, i32* %P, i64 %idxprom4
  store i32 0, i32* %arrayidx5
  %add6 = add nsw i32 %n, 3
  %idxprom7 = sext i32 %add6 to i64
  %arrayidx8 = getelementptr inbounds i32, i32* %P, i64 %idxprom7
  store i32 0, i32* %arrayidx8
  ret void
}

; CHECK-LABEL: Sturb_zero
; CHECK: sturh wzr
; CHECK-STRICT-LABEL: Sturb_zero
; CHECK-STRICT: sturb wzr
; CHECK-STRICT: sturb wzr
define void @Sturb_zero(i8* nocapture %P, i32 %n) #0 {
entry:
  %sub = add nsw i32 %n, -2
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i8, i8* %P, i64 %idxprom
  store i8 0, i8* %arrayidx
  %sub2= add nsw i32 %n, -1
  %idxprom1 = sext i32 %sub2 to i64
  %arrayidx2 = getelementptr inbounds i8, i8* %P, i64 %idxprom1
  store i8 0, i8* %arrayidx2
  ret void
}

; CHECK-LABEL: Sturh_zero
; CHECK: stur wzr
; CHECK-STRICT-LABEL: Sturh_zero
; CHECK-STRICT: sturh wzr
; CHECK-STRICT: sturh wzr
define void @Sturh_zero(i16* nocapture %P, i32 %n) {
entry:
  %sub = add nsw i32 %n, -2
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i16, i16* %P, i64 %idxprom
  store i16 0, i16* %arrayidx
  %sub1 = add nsw i32 %n, -3
  %idxprom2 = sext i32 %sub1 to i64
  %arrayidx3 = getelementptr inbounds i16, i16* %P, i64 %idxprom2
  store i16 0, i16* %arrayidx3
  ret void
}

; CHECK-LABEL: Sturh_zero_4
; CHECK: stur xzr
; CHECK-STRICT-LABEL: Sturh_zero_4
; CHECK-STRICT: sturh wzr
; CHECK-STRICT: sturh wzr
; CHECK-STRICT: sturh wzr
; CHECK-STRICT: sturh wzr
define void @Sturh_zero_4(i16* nocapture %P, i32 %n) {
entry:
  %sub = add nsw i32 %n, -3
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i16, i16* %P, i64 %idxprom
  store i16 0, i16* %arrayidx
  %sub1 = add nsw i32 %n, -4
  %idxprom2 = sext i32 %sub1 to i64
  %arrayidx3 = getelementptr inbounds i16, i16* %P, i64 %idxprom2
  store i16 0, i16* %arrayidx3
  %sub4 = add nsw i32 %n, -2
  %idxprom5 = sext i32 %sub4 to i64
  %arrayidx6 = getelementptr inbounds i16, i16* %P, i64 %idxprom5
  store i16 0, i16* %arrayidx6
  %sub7 = add nsw i32 %n, -1
  %idxprom8 = sext i32 %sub7 to i64
  %arrayidx9 = getelementptr inbounds i16, i16* %P, i64 %idxprom8
  store i16 0, i16* %arrayidx9
  ret void
}

; CHECK-LABEL: Sturw_zero
; CHECK: stur xzr
; CHECK-STRICT-LABEL: Sturw_zero
; CHECK-STRICT: stp wzr, wzr
define void @Sturw_zero(i32* nocapture %P, i32 %n) {
entry:
  %sub = add nsw i32 %n, -3
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i32, i32* %P, i64 %idxprom
  store i32 0, i32* %arrayidx
  %sub1 = add nsw i32 %n, -4
  %idxprom2 = sext i32 %sub1 to i64
  %arrayidx3 = getelementptr inbounds i32, i32* %P, i64 %idxprom2
  store i32 0, i32* %arrayidx3
  ret void
}

; CHECK-LABEL: Sturw_zero_4
; CHECK: stp xzr, xzr
; CHECK-STRICT-LABEL: Sturw_zero_4
; CHECK-STRICT: stp wzr, wzr
; CHECK-STRICT: stp wzr, wzr
define void @Sturw_zero_4(i32* nocapture %P, i32 %n) {
entry:
  %sub = add nsw i32 %n, -3
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i32, i32* %P, i64 %idxprom
  store i32 0, i32* %arrayidx
  %sub1 = add nsw i32 %n, -4
  %idxprom2 = sext i32 %sub1 to i64
  %arrayidx3 = getelementptr inbounds i32, i32* %P, i64 %idxprom2
  store i32 0, i32* %arrayidx3
  %sub4 = add nsw i32 %n, -2
  %idxprom5 = sext i32 %sub4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* %P, i64 %idxprom5
  store i32 0, i32* %arrayidx6
  %sub7 = add nsw i32 %n, -1
  %idxprom8 = sext i32 %sub7 to i64
  %arrayidx9 = getelementptr inbounds i32, i32* %P, i64 %idxprom8
  store i32 0, i32* %arrayidx9
  ret void
}

