; RUN: llc < %s -mtriple aarch64 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple aarch64 -mattr=+strict-align -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-STRICT

; CHECK-LABEL: Strh_zero
; CHECK: str wzr
; CHECK-STRICT-LABEL: Strh_zero
; CHECK-STRICT: strh wzr
; CHECK-STRICT: strh wzr
define void @Strh_zero(ptr nocapture %P, i32 %n) {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i16, ptr %P, i64 %idxprom
  store i16 0, ptr %arrayidx
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i16, ptr %P, i64 %idxprom1
  store i16 0, ptr %arrayidx2
  ret void
}

; CHECK-LABEL: Strh_zero_4
; CHECK: str xzr
; CHECK-STRICT-LABEL: Strh_zero_4
; CHECK-STRICT: strh wzr
; CHECK-STRICT: strh wzr
; CHECK-STRICT: strh wzr
; CHECK-STRICT: strh wzr
define void @Strh_zero_4(ptr nocapture %P, i32 %n) {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i16, ptr %P, i64 %idxprom
  store i16 0, ptr %arrayidx
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i16, ptr %P, i64 %idxprom1
  store i16 0, ptr %arrayidx2
  %add3 = add nsw i32 %n, 2
  %idxprom4 = sext i32 %add3 to i64
  %arrayidx5 = getelementptr inbounds i16, ptr %P, i64 %idxprom4
  store i16 0, ptr %arrayidx5
  %add6 = add nsw i32 %n, 3
  %idxprom7 = sext i32 %add6 to i64
  %arrayidx8 = getelementptr inbounds i16, ptr %P, i64 %idxprom7
  store i16 0, ptr %arrayidx8
  ret void
}

; CHECK-LABEL: Strw_zero
; CHECK: str xzr
; CHECK-STRICT-LABEL: Strw_zero
; CHECK-STRICT: stp wzr, wzr
define void @Strw_zero(ptr nocapture %P, i32 %n) {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i32, ptr %P, i64 %idxprom
  store i32 0, ptr %arrayidx
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32, ptr %P, i64 %idxprom1
  store i32 0, ptr %arrayidx2
  ret void
}

; CHECK-LABEL: Strw_zero_nonzero
; CHECK: stp wzr, w1
define void @Strw_zero_nonzero(ptr nocapture %P, i32 %n)  {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i32, ptr %P, i64 %idxprom
  store i32 0, ptr %arrayidx
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32, ptr %P, i64 %idxprom1
  store i32 %n, ptr %arrayidx2
  ret void
}

; CHECK-LABEL: Strw_zero_4
; CHECK: stp xzr, xzr
; CHECK-STRICT-LABEL: Strw_zero_4
; CHECK-STRICT: stp wzr, wzr
; CHECK-STRICT: stp wzr, wzr
define void @Strw_zero_4(ptr nocapture %P, i32 %n) {
entry:
  %idxprom = sext i32 %n to i64
  %arrayidx = getelementptr inbounds i32, ptr %P, i64 %idxprom
  store i32 0, ptr %arrayidx
  %add = add nsw i32 %n, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32, ptr %P, i64 %idxprom1
  store i32 0, ptr %arrayidx2
  %add3 = add nsw i32 %n, 2
  %idxprom4 = sext i32 %add3 to i64
  %arrayidx5 = getelementptr inbounds i32, ptr %P, i64 %idxprom4
  store i32 0, ptr %arrayidx5
  %add6 = add nsw i32 %n, 3
  %idxprom7 = sext i32 %add6 to i64
  %arrayidx8 = getelementptr inbounds i32, ptr %P, i64 %idxprom7
  store i32 0, ptr %arrayidx8
  ret void
}

; CHECK-LABEL: Sturb_zero
; CHECK: sturh wzr
; CHECK-STRICT-LABEL: Sturb_zero
; CHECK-STRICT: sturb wzr
; CHECK-STRICT: sturb wzr
define void @Sturb_zero(ptr nocapture %P, i32 %n) #0 {
entry:
  %sub = add nsw i32 %n, -2
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i8, ptr %P, i64 %idxprom
  store i8 0, ptr %arrayidx
  %sub2= add nsw i32 %n, -1
  %idxprom1 = sext i32 %sub2 to i64
  %arrayidx2 = getelementptr inbounds i8, ptr %P, i64 %idxprom1
  store i8 0, ptr %arrayidx2
  ret void
}

; CHECK-LABEL: Sturh_zero
; CHECK: stur wzr
; CHECK-STRICT-LABEL: Sturh_zero
; CHECK-STRICT: sturh wzr
; CHECK-STRICT: sturh wzr
define void @Sturh_zero(ptr nocapture %P, i32 %n) {
entry:
  %sub = add nsw i32 %n, -2
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i16, ptr %P, i64 %idxprom
  store i16 0, ptr %arrayidx
  %sub1 = add nsw i32 %n, -3
  %idxprom2 = sext i32 %sub1 to i64
  %arrayidx3 = getelementptr inbounds i16, ptr %P, i64 %idxprom2
  store i16 0, ptr %arrayidx3
  ret void
}

; CHECK-LABEL: Sturh_zero_4
; CHECK: stur xzr
; CHECK-STRICT-LABEL: Sturh_zero_4
; CHECK-STRICT: sturh wzr
; CHECK-STRICT: sturh wzr
; CHECK-STRICT: sturh wzr
; CHECK-STRICT: sturh wzr
define void @Sturh_zero_4(ptr nocapture %P, i32 %n) {
entry:
  %sub = add nsw i32 %n, -3
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i16, ptr %P, i64 %idxprom
  store i16 0, ptr %arrayidx
  %sub1 = add nsw i32 %n, -4
  %idxprom2 = sext i32 %sub1 to i64
  %arrayidx3 = getelementptr inbounds i16, ptr %P, i64 %idxprom2
  store i16 0, ptr %arrayidx3
  %sub4 = add nsw i32 %n, -2
  %idxprom5 = sext i32 %sub4 to i64
  %arrayidx6 = getelementptr inbounds i16, ptr %P, i64 %idxprom5
  store i16 0, ptr %arrayidx6
  %sub7 = add nsw i32 %n, -1
  %idxprom8 = sext i32 %sub7 to i64
  %arrayidx9 = getelementptr inbounds i16, ptr %P, i64 %idxprom8
  store i16 0, ptr %arrayidx9
  ret void
}

; CHECK-LABEL: Sturw_zero
; CHECK: stur xzr
; CHECK-STRICT-LABEL: Sturw_zero
; CHECK-STRICT: stp wzr, wzr
define void @Sturw_zero(ptr nocapture %P, i32 %n) {
entry:
  %sub = add nsw i32 %n, -3
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i32, ptr %P, i64 %idxprom
  store i32 0, ptr %arrayidx
  %sub1 = add nsw i32 %n, -4
  %idxprom2 = sext i32 %sub1 to i64
  %arrayidx3 = getelementptr inbounds i32, ptr %P, i64 %idxprom2
  store i32 0, ptr %arrayidx3
  ret void
}

; CHECK-LABEL: Sturw_zero_4
; CHECK: stp xzr, xzr
; CHECK-STRICT-LABEL: Sturw_zero_4
; CHECK-STRICT: stp wzr, wzr
; CHECK-STRICT: stp wzr, wzr
define void @Sturw_zero_4(ptr nocapture %P, i32 %n) {
entry:
  %sub = add nsw i32 %n, -3
  %idxprom = sext i32 %sub to i64
  %arrayidx = getelementptr inbounds i32, ptr %P, i64 %idxprom
  store i32 0, ptr %arrayidx
  %sub1 = add nsw i32 %n, -4
  %idxprom2 = sext i32 %sub1 to i64
  %arrayidx3 = getelementptr inbounds i32, ptr %P, i64 %idxprom2
  store i32 0, ptr %arrayidx3
  %sub4 = add nsw i32 %n, -2
  %idxprom5 = sext i32 %sub4 to i64
  %arrayidx6 = getelementptr inbounds i32, ptr %P, i64 %idxprom5
  store i32 0, ptr %arrayidx6
  %sub7 = add nsw i32 %n, -1
  %idxprom8 = sext i32 %sub7 to i64
  %arrayidx9 = getelementptr inbounds i32, ptr %P, i64 %idxprom8
  store i32 0, ptr %arrayidx9
  ret void
}

