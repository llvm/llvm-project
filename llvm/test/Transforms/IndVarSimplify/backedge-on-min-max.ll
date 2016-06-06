; RUN: opt < %s -indvars -S | FileCheck %s
; RUN: opt -lcssa -loop-simplify -S < %s | opt -S -passes='require<targetir>,require<scalar-evolution>,require<domtree>,loop(indvars)'

;; --- signed ---

define void @min.signed.1(i32* %a, i32 %a_len, i32 %n) {
; CHECK-LABEL: @min.signed.1
 entry:
  %smin.cmp = icmp slt i32 %a_len, %n
  %smin = select i1 %smin.cmp, i32 %a_len, i32 %n
  %entry.cond = icmp slt i32 0, %smin
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp slt i32 %idx, %a_len
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp slt i32 %idx.inc, %smin
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @min.signed.2(i32* %a, i32 %a_len, i32 %n) {
; CHECK-LABEL: @min.signed.2
 entry:
  %smin.cmp = icmp slt i32 %a_len, %n
  %smin = select i1 %smin.cmp, i32 %a_len, i32 %n
  %entry.cond = icmp slt i32 0, %smin
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp sgt i32 %a_len, %idx
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp slt i32 %idx.inc, %smin
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @min.signed.3(i32* %a, i32 %n) {
; CHECK-LABEL: @min.signed.3
 entry:
  %smin.cmp = icmp slt i32 42, %n
  %smin = select i1 %smin.cmp, i32 42, i32 %n
  %entry.cond = icmp slt i32 0, %smin
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp slt i32 %idx, 42
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp slt i32 %idx.inc, %smin
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @min.signed.4(i32* %a, i32 %n) {
; CHECK-LABEL: @min.signed.4
 entry:
  %smin.cmp = icmp slt i32 42, %n
  %smin = select i1 %smin.cmp, i32 42, i32 %n
  %entry.cond = icmp slt i32 0, %smin
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp sgt i32 42, %idx
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp slt i32 %idx.inc, %smin
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @max.signed.1(i32* %a, i32 %a_len, i32 %n) {
; CHECK-LABEL: @max.signed.1
 entry:
  %smax.cmp = icmp sgt i32 %a_len, %n
  %smax = select i1 %smax.cmp, i32 %a_len, i32 %n
  %entry.cond = icmp sgt i32 0, %smax
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp sgt i32 %idx, %a_len
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp sgt i32 %idx.inc, %smax
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @max.signed.2(i32* %a, i32 %a_len, i32 %n) {
; CHECK-LABEL: @max.signed.2
 entry:
  %smax.cmp = icmp sgt i32 %a_len, %n
  %smax = select i1 %smax.cmp, i32 %a_len, i32 %n
  %entry.cond = icmp sgt i32 0, %smax
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp slt i32 %a_len, %idx
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp sgt i32 %idx.inc, %smax
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @max.signed.3(i32* %a, i32 %n, i32 %init) {
; CHECK-LABEL: @max.signed.3
 entry:
  %smax.cmp = icmp sgt i32 42, %n
  %smax = select i1 %smax.cmp, i32 42, i32 %n
  %entry.cond = icmp sgt i32 %init, %smax
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ %init, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp sgt i32 %idx, 42
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp sgt i32 %idx.inc, %smax
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @max.signed.4(i32* %a, i32 %n, i32 %init) {
; CHECK-LABEL: @max.signed.4
 entry:
  %smax.cmp = icmp sgt i32 42, %n
  %smax = select i1 %smax.cmp, i32 42, i32 %n
  %entry.cond = icmp sgt i32 %init, %smax
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ %init, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp slt i32 42, %idx
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp sgt i32 %idx.inc, %smax
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

;; --- unsigned ---

define void @min.unsigned.1(i32* %a, i32 %a_len, i32 %n) {
; CHECK-LABEL: @min.unsigned.1
 entry:
  %umin.cmp = icmp ult i32 %a_len, %n
  %umin = select i1 %umin.cmp, i32 %a_len, i32 %n
  %entry.cond = icmp ult i32 5, %umin
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 5, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp ult i32 %idx, %a_len
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp ult i32 %idx.inc, %umin
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @min.unsigned.2(i32* %a, i32 %a_len, i32 %n) {
; CHECK-LABEL: @min.unsigned.2
 entry:
  %umin.cmp = icmp ult i32 %a_len, %n
  %umin = select i1 %umin.cmp, i32 %a_len, i32 %n
  %entry.cond = icmp ult i32 5, %umin
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 5, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp ugt i32 %a_len, %idx
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp ult i32 %idx.inc, %umin
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @min.unsigned.3(i32* %a, i32 %n) {
; CHECK-LABEL: @min.unsigned.3
 entry:
  %umin.cmp = icmp ult i32 42, %n
  %umin = select i1 %umin.cmp, i32 42, i32 %n
  %entry.cond = icmp ult i32 5, %umin
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 5, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp ult i32 %idx, 42
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp ult i32 %idx.inc, %umin
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @min.unsigned.4(i32* %a, i32 %n) {
; CHECK-LABEL: @min.unsigned.4
 entry:
  %umin.cmp = icmp ult i32 42, %n
  %umin = select i1 %umin.cmp, i32 42, i32 %n
  %entry.cond = icmp ult i32 5, %umin
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 5, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp ugt i32 42, %idx
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp ult i32 %idx.inc, %umin
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @max.unsigned.1(i32* %a, i32 %a_len, i32 %n) {
; CHECK-LABEL: @max.unsigned.1
 entry:
  %umax.cmp = icmp ugt i32 %a_len, %n
  %umax = select i1 %umax.cmp, i32 %a_len, i32 %n
  %entry.cond = icmp ugt i32 5, %umax
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 5, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp ugt i32 %idx, %a_len
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp ugt i32 %idx.inc, %umax
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @max.unsigned.2(i32* %a, i32 %a_len, i32 %n) {
; CHECK-LABEL: @max.unsigned.2
 entry:
  %umax.cmp = icmp ugt i32 %a_len, %n
  %umax = select i1 %umax.cmp, i32 %a_len, i32 %n
  %entry.cond = icmp ugt i32 5, %umax
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ 5, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp ult i32 %a_len, %idx
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp ugt i32 %idx.inc, %umax
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @max.unsigned.3(i32* %a, i32 %n, i32 %init) {
; CHECK-LABEL: @max.unsigned.3
 entry:
  %umax.cmp = icmp ugt i32 42, %n
  %umax = select i1 %umax.cmp, i32 42, i32 %n
  %entry.cond = icmp ugt i32 %init, %umax
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ %init, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp ugt i32 %idx, 42
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp ugt i32 %idx.inc, %umax
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}

define void @max.unsigned.4(i32* %a, i32 %n, i32 %init) {
; CHECK-LABEL: @max.unsigned.4
 entry:
  %umax.cmp = icmp ugt i32 42, %n
  %umax = select i1 %umax.cmp, i32 42, i32 %n
  %entry.cond = icmp ugt i32 %init, %umax
  br i1 %entry.cond, label %loop, label %exit

 loop:
  %idx = phi i32 [ %init, %entry ], [ %idx.inc, %latch ]
  %idx.inc = add i32 %idx, 1
  %in.bounds = icmp ult i32 42, %idx
  br i1 %in.bounds, label %ok, label %latch
; CHECK: br i1 true, label %ok, label %latch

 ok:
  %addr = getelementptr i32, i32* %a, i32 %idx
  store i32 %idx, i32* %addr
  br label %latch

 latch:
  %be.cond = icmp ugt i32 %idx.inc, %umax
  br i1 %be.cond, label %loop, label %exit

 exit:
  ret void
}
