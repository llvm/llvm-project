; RUN: opt -arm-codegenprepare -arm-disable-cgp=false -mtriple=armv8 -verify %s -S -o - | FileCheck %s

; CHECK: clear_structures
define i32 @clear_structures(i8* nocapture readonly %fmt, [1 x i32] %ap.coerce, i8* %out, void (i32, i8*)* nocapture %write) {
entry:
  br label %while.cond.outer

while.cond.outer:
  %fmt.addr.0.ph = phi i8* [ %fmt, %entry ], [ %fmt.addr.3, %while.cond.outer.backedge ]
  %0 = load i8, i8* %fmt.addr.0.ph, align 1
  br label %while.cond

while.cond:
  switch i8 %0, label %while.cond [
    i8 0, label %while.end48
    i8 37, label %while.cond2
  ]

while.cond2:
  %flags.0 = phi i32 [ %or, %while.cond2 ], [ 0, %while.cond ]
  %fmt.addr.0.pn = phi i8* [ %fmt.addr.1, %while.cond2 ], [ %fmt.addr.0.ph, %while.cond ]
  %fmt.addr.1 = getelementptr inbounds i8, i8* %fmt.addr.0.pn, i32 1
  %1 = load i8, i8* %fmt.addr.1, align 1
  ; CHECK: add i8 [[LOAD:%[^ ]+]], -32
  %sub = add i8 %1, -32
  %conv6 = zext i8 %sub to i32
  %shl = shl i32 1, %conv6
  %and = and i32 %shl, 75785
  %tobool7 = icmp eq i32 %and, 0
  %or = or i32 %shl, %flags.0
  br i1 %tobool7, label %while.cond10.preheader, label %while.cond2

while.cond10.preheader:
  ; CHECK: [[ADD:%[^ ]+]] = add i8 [[LOAD]], -48
  ; CHECK: icmp ult i8 [[ADD]], 10
  %.off = add i8 %1, -48
  %2 = icmp ult i8 %.off, 10
  br i1 %2, label %while.cond10, label %while.end18.split

while.cond10:
  br label %while.cond10

while.end18.split:
  %cmp20 = icmp eq i8 %1, 46
  br i1 %cmp20, label %if.then22, label %cond.end

if.then22:
  %incdec.ptr23 = getelementptr inbounds i8, i8* %fmt.addr.0.pn, i32 2
  %.pr74 = load i8, i8* %incdec.ptr23, align 1
  ; CHECK: [[LOAD2:[^ ]+]] = load i8, i8*
  ; CHECK: [[ZEXT:[^ ]+]] = zext i8 [[LOAD2]] to i32
  ; CHECK: sub i32 [[ZEXT]], 48
  %.pr74.off = add i8 %.pr74, -48
  %3 = icmp ult i8 %.pr74.off, 10
  br i1 %3, label %while.cond24, label %cond.end

while.cond24:
  br label %while.cond24

cond.end:
  %fmt.addr.3 = phi i8* [ %fmt.addr.1, %while.end18.split ], [ %incdec.ptr23, %if.then22 ]
  %and39 = and i32 %flags.0, 2048
  %tobool40 = icmp eq i32 %and39, 0
  br i1 %tobool40, label %while.cond.outer.backedge, label %if.then43

while.cond.outer.backedge:
  br label %while.cond.outer

if.then43:
  tail call void %write(i32 43, i8* %out) #1
  br label %while.cond.outer.backedge

while.end48:
  ret i32 undef
}
