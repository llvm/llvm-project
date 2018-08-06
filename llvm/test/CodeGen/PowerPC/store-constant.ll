; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs | FileCheck %s

@CVal = external local_unnamed_addr global i8, align 1
@SVal = external local_unnamed_addr global i16, align 2
@IVal = external local_unnamed_addr global i32, align 4
@LVal = external local_unnamed_addr global i64, align 8
@USVal = external local_unnamed_addr global i16, align 2
@arr = external local_unnamed_addr global i64*, align 8
@arri = external local_unnamed_addr global i32*, align 8

; Test the same constant can be used by different stores.

%struct.S = type { i64, i8, i16, i32 }

define void @foo(%struct.S* %p) {
  %l4 = bitcast %struct.S* %p to i64*
  store i64 0, i64* %l4, align 8
  %c = getelementptr %struct.S, %struct.S* %p, i64 0, i32 1
  store i8 0, i8* %c, align 8
  %s = getelementptr %struct.S, %struct.S* %p, i64 0, i32 2
  store i16 0, i16* %s, align 2
  %i = getelementptr %struct.S, %struct.S* %p, i64 0, i32 3
  store i32 0, i32* %i, align 4
  ret void

; CHECK-LABEL: @foo
; CHECK:       li 4, 0
; CHECK:       stb 4, 8(3)
; CHECK:       std 4, 0(3)
; CHECK:       sth 4, 10(3)
; CHECK:       stw 4, 12(3)
}

define void @bar(%struct.S* %p) {
  %i = getelementptr %struct.S, %struct.S* %p, i64 0, i32 3
  store i32 2, i32* %i, align 4
  %s = getelementptr %struct.S, %struct.S* %p, i64 0, i32 2
  store i16 2, i16* %s, align 2
  %c = getelementptr %struct.S, %struct.S* %p, i64 0, i32 1
  store i8 2, i8* %c, align 8
  %l4 = bitcast %struct.S* %p to i64*
  store i64 2, i64* %l4, align 8
  ret void

; CHECK-LABEL: @bar
; CHECK:       li 4, 2
; CHECK-DAG:       stw 4, 12(3)
; CHECK-DAG:       sth 4, 10(3)
; CHECK-DAG:       std 4, 0(3)
; CHECK-DAG:       stb 4, 8(3)
}

; Function Attrs: norecurse nounwind
define void @setSmallNeg() {
entry:
  store i8 -7, i8* @CVal, align 1
  store i16 -7, i16* @SVal, align 2
  store i32 -7, i32* @IVal, align 4
  store i64 -7, i64* @LVal, align 8
  ret void
; CHECK-LABEL: setSmallNeg
; CHECK: li 7, -7
; CHECK-DAG: stb 7,
; CHECK-DAG: sth 7,
; CHECK-DAG: stw 7,
; CHECK-DAG: std 7,
}

; Function Attrs: norecurse nounwind
define void @setSmallPos() {
entry:
  store i8 8, i8* @CVal, align 1
  store i16 8, i16* @SVal, align 2
  store i32 8, i32* @IVal, align 4
  store i64 8, i64* @LVal, align 8
  ret void
; CHECK-LABEL: setSmallPos
; CHECK: li 7, 8
; CHECK-DAG: stb 7,
; CHECK-DAG: sth 7,
; CHECK-DAG: stw 7,
; CHECK-DAG: std 7,
}

; Function Attrs: norecurse nounwind
define void @setMaxNeg() {
entry:
  store i16 -32768, i16* @SVal, align 2
  store i32 -32768, i32* @IVal, align 4
  store i64 -32768, i64* @LVal, align 8
  ret void
; CHECK-LABEL: setMaxNeg
; CHECK: li 6, -32768
; CHECK-DAG: sth 6,
; CHECK-DAG: stw 6,
; CHECK-DAG: std 6,
}

; Function Attrs: norecurse nounwind
define void @setMaxPos() {
entry:
  store i16 32767, i16* @SVal, align 2
  store i32 32767, i32* @IVal, align 4
  store i64 32767, i64* @LVal, align 8
  ret void
; CHECK-LABEL: setMaxPos
; CHECK: li 6, 32767
; CHECK-DAG: sth 6,
; CHECK-DAG: stw 6,
; CHECK-DAG: std 6,
}

; Function Attrs: norecurse nounwind
define void @setExcessiveNeg() {
entry:
  store i32 -32769, i32* @IVal, align 4
  store i64 -32769, i64* @LVal, align 8
  ret void
; CHECK-LABEL: setExcessiveNeg
; CHECK: lis 5, -1
; CHECK: ori 5, 5, 32767
; CHECK-DAG: stw 5,
; CHECK-DAG: std 5,
}

; Function Attrs: norecurse nounwind
define void @setExcessivePos() {
entry:
  store i16 -32768, i16* @USVal, align 2
  store i32 32768, i32* @IVal, align 4
  store i64 32768, i64* @LVal, align 8
  ret void
; CHECK-LABEL: setExcessivePos
; CHECK: li 6, 0
; CHECK: ori 6, 6, 32768
; CHECK-DAG: sth 6,
; CHECK-DAG: stw 6,
; CHECK-DAG: std 6,
}

define void @SetArr(i32 signext %Len) {
entry:
  %cmp7 = icmp sgt i32 %Len, 0
  br i1 %cmp7, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %0 = load i64*, i64** @arr, align 8
  %1 = load i32*, i32** @arri, align 8
  %wide.trip.count = zext i32 %Len to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %0, i64 %indvars.iv
  store i64 -7, i64* %arrayidx, align 8
  %arrayidx2 = getelementptr inbounds i32, i32* %1, i64 %indvars.iv
  store i32 -7, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
; CHECK-LABEL: SetArr
; CHECK: li 5, -7
; CHECK: stdu 5, 8(3)
; CHECK: stwu 5, 4(4)
}

define void @setSameValDiffSizeCI() {
entry:
  store i32 255, i32* @IVal, align 4
  store i8 -1, i8* @CVal, align 1
  ret void
; CHECK-LABEL: setSameValDiffSizeCI
; CHECK: li 5, 255
; CHECK-DAG: stb 5,
; CHECK-DAG: stw 5,
}

define void @setSameValDiffSizeSI() {
entry:
  store i32 65535, i32* @IVal, align 4
  store i16 -1, i16* @SVal, align 2
  ret void
; CHECK-LABEL: setSameValDiffSizeSI
; CHECK: li 5, 0
; CHECK: ori 5, 5, 65535
; CHECK-DAG: sth 5,
; CHECK-DAG: stw 5,
}
