; RUN: opt -S -codegenprepare -mtriple=r600-unknown-unknown -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=ALL %s
; RUN: opt -S -codegenprepare -mtriple=r600-unknown-unknown -mcpu=tonga < %s | FileCheck -check-prefix=SI -check-prefix=ALL %s


define i64 @test1(i64 %A) {
; ALL-LABEL: @test1(
; SI: [[CTLZ:%[A-Za-z0-9]+]] = call i64 @llvm.ctlz.i64(i64 %A, i1 false)
; SI-NEXT: ret i64 [[CTLZ]]
entry:
  %tobool = icmp eq i64 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.ctlz.i64(i64 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i64 [ %0, %cond.true ], [ 64, %entry ]
  ret i64 %cond
}


define i32 @test2(i32 %A) {
; ALL-LABEL: @test2(
; SI: [[CTLZ:%[A-Za-z0-9]+]] = call i32 @llvm.ctlz.i32(i32 %A, i1 false)
; SI-NEXT: ret i32 [[CTLZ]]
entry:
  %tobool = icmp eq i32 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.ctlz.i32(i32 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %0, %cond.true ], [ 32, %entry ]
  ret i32 %cond
}


define signext i16 @test3(i16 signext %A) {
; ALL-LABEL: @test3(
; SI: [[CTLZ:%[A-Za-z0-9]+]] = call i16 @llvm.ctlz.i16(i16 %A, i1 false)
; SI-NEXT: ret i16 [[CTLZ]]
entry:
  %tobool = icmp eq i16 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i16 @llvm.ctlz.i16(i16 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %0, %cond.true ], [ 16, %entry ]
  ret i16 %cond
}


define i64 @test1b(i64 %A) {
; ALL-LABEL: @test1b(
; SI: [[CTTZ:%[A-Za-z0-9]+]] = call i64 @llvm.cttz.i64(i64 %A, i1 false)
; SI-NEXT: ret i64 [[CTTZ]]
entry:
  %tobool = icmp eq i64 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.cttz.i64(i64 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i64 [ %0, %cond.true ], [ 64, %entry ]
  ret i64 %cond
}


define i32 @test2b(i32 %A) {
; ALL-LABEL: @test2b(
; SI: [[CTTZ:%[A-Za-z0-9]+]] = call i32 @llvm.cttz.i32(i32 %A, i1 false)
; SI-NEXT: ret i32 [[CTTZ]]
entry:
  %tobool = icmp eq i32 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.cttz.i32(i32 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %0, %cond.true ], [ 32, %entry ]
  ret i32 %cond
}


define signext i16 @test3b(i16 signext %A) {
; ALL-LABEL: @test3b(
; SI: [[CTTZ:%[A-Za-z0-9]+]] = call i16 @llvm.cttz.i16(i16 %A, i1 false)
; SI-NEXT: ret i16 [[CTTZ]]
entry:
  %tobool = icmp eq i16 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i16 @llvm.cttz.i16(i16 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %0, %cond.true ], [ 16, %entry ]
  ret i16 %cond
}


define i64 @test1c(i64 %A) {
; ALL-LABEL: @test1c(
; ALL: icmp eq i64 %A, 0
; ALL: call i64 @llvm.ctlz.i64(i64 %A, i1 true)
entry:
  %tobool = icmp eq i64 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.ctlz.i64(i64 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i64 [ %0, %cond.true ], [ 63, %entry ]
  ret i64 %cond
}

define i32 @test2c(i32 %A) {
; ALL-LABEL: @test2c(
; ALL: icmp eq i32 %A, 0
; ALL: call i32 @llvm.ctlz.i32(i32 %A, i1 true)
entry:
  %tobool = icmp eq i32 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.ctlz.i32(i32 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %0, %cond.true ], [ 31, %entry ]
  ret i32 %cond
}


define signext i16 @test3c(i16 signext %A) {
; ALL-LABEL: @test3c(
; ALL: icmp eq i16 %A, 0
; ALL: call i16 @llvm.ctlz.i16(i16 %A, i1 true)
entry:
  %tobool = icmp eq i16 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i16 @llvm.ctlz.i16(i16 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %0, %cond.true ], [ 15, %entry ]
  ret i16 %cond
}


define i64 @test1d(i64 %A) {
; ALL-LABEL: @test1d(
; ALL: icmp eq i64 %A, 0
; ALL: call i64 @llvm.cttz.i64(i64 %A, i1 true)
entry:
  %tobool = icmp eq i64 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i64 @llvm.cttz.i64(i64 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i64 [ %0, %cond.true ], [ 63, %entry ]
  ret i64 %cond
}


define i32 @test2d(i32 %A) {
; ALL-LABEL: @test2d(
; ALL: icmp eq i32 %A, 0
; ALL: call i32 @llvm.cttz.i32(i32 %A, i1 true)
entry:
  %tobool = icmp eq i32 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i32 @llvm.cttz.i32(i32 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %0, %cond.true ], [ 31, %entry ]
  ret i32 %cond
}


define signext i16 @test3d(i16 signext %A) {
; ALL-LABEL: @test3d(
; ALL: icmp eq i16 %A, 0
; ALL: call i16 @llvm.cttz.i16(i16 %A, i1 true)
entry:
  %tobool = icmp eq i16 %A, 0
  br i1 %tobool, label %cond.end, label %cond.true

cond.true:                                        ; preds = %entry
  %0 = tail call i16 @llvm.cttz.i16(i16 %A, i1 true)
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i16 [ %0, %cond.true ], [ 15, %entry ]
  ret i16 %cond
}


declare i64 @llvm.ctlz.i64(i64, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i16 @llvm.ctlz.i16(i16, i1)
declare i64 @llvm.cttz.i64(i64, i1)
declare i32 @llvm.cttz.i32(i32, i1)
declare i16 @llvm.cttz.i16(i16, i1)
