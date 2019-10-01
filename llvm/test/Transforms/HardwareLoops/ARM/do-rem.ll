; RUN: opt -mtriple=thumbv8.1m.main-arm-none-eabi -hardware-loops -disable-arm-loloops=false %s -S -o - | FileCheck %s

@g = common local_unnamed_addr global i32* null, align 4

; CHECK-LABEL: do_with_i32_urem
; CHECK: while.body.preheader:
; CHECK: call void @llvm.set.loop.iterations.i32(i32 %n)
; CHECK-NEXT: br label %while.body

; CHECK: [[REM:%[^ ]+]] = phi i32 [ %n, %while.body.preheader ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 [[REM]], i32 1)
; CHECK: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK: br i1 [[CMP]], label %while.body, label %while.end.loopexit

define i32 @do_with_i32_urem(i32 %n) {
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %i.09 = phi i32 [ %inc1, %while.body ], [ 0, %while.body.preheader ]
  %res.08 = phi i32 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %rem = urem i32 %i.09, 5
  %add = add i32 %rem, %res.08
  %inc1 = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: do_with_i32_srem
; CHECK: while.body.preheader:
; CHECK: call void @llvm.set.loop.iterations.i32(i32 %n)
; CHECK-NEXT: br label %while.body

; CHECK: [[REM:%[^ ]+]] = phi i32 [ %n, %while.body.preheader ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 [[REM]], i32 1)
; CHECK: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK: br i1 [[CMP]], label %while.body, label %while.end.loopexit

define i32 @do_with_i32_srem(i32 %n) {
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %i.09 = phi i32 [ %inc1, %while.body ], [ 0, %while.body.preheader ]
  %res.08 = phi i32 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %rem = srem i32 %i.09, 5
  %add = sub i32 %rem, %res.08
  %inc1 = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: do_with_i32_udiv
; CHECK: while.body.preheader:
; CHECK: call void @llvm.set.loop.iterations.i32(i32 %n)
; CHECK-NEXT: br label %while.body

; CHECK: [[REM:%[^ ]+]] = phi i32 [ %n, %while.body.preheader ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 [[REM]], i32 1)
; CHECK: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK: br i1 [[CMP]], label %while.body, label %while.end.loopexit

define i32 @do_with_i32_udiv(i32 %n) {
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %i.09 = phi i32 [ %inc1, %while.body ], [ 0, %while.body.preheader ]
  %res.08 = phi i32 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %rem = udiv i32 %i.09, 5
  %add = add i32 %rem, %res.08
  %inc1 = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: do_with_i32_sdiv
; CHECK: while.body.preheader:
; CHECK: call void @llvm.set.loop.iterations.i32(i32 %n)
; CHECK-NEXT: br label %while.body

; CHECK: [[REM:%[^ ]+]] = phi i32 [ %n, %while.body.preheader ], [ [[LOOP_DEC:%[^ ]+]], %while.body ]
; CHECK: [[LOOP_DEC]] = call i32 @llvm.loop.decrement.reg.i32.i32.i32(i32 [[REM]], i32 1)
; CHECK: [[CMP:%[^ ]+]] = icmp ne i32 [[LOOP_DEC]], 0
; CHECK: br i1 [[CMP]], label %while.body, label %while.end.loopexit

define i32 @do_with_i32_sdiv(i32 %n) {
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %i.09 = phi i32 [ %inc1, %while.body ], [ 0, %while.body.preheader ]
  %res.08 = phi i32 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %rem = sdiv i32 %i.09, 5
  %add = sub i32 %rem, %res.08
  %inc1 = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i32 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i32 %res.0.lcssa
}

; CHECK-LABEL: do_with_i64_urem
; CHECK-NOT: llvm.set.loop.iterations
; CHECK-NOT: llvm.loop.decrement
define i64 @do_with_i64_urem(i32 %n) {
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %i.09 = phi i32 [ %inc1, %while.body ], [ 0, %while.body.preheader ]
  %res.08 = phi i64 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %conv = zext i32 %i.09 to i64
  %rem = urem i64 %conv, 5
  %add = add i64 %rem, %res.08
  %inc1 = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i64 %res.0.lcssa
}

; CHECK-LABEL: do_with_i64_srem
; CHECK-NOT: llvm.set.loop.iterations
; CHECK-NOT: llvm.loop.decrement
define i64 @do_with_i64_srem(i32 %n) {
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %i.09 = phi i32 [ %inc1, %while.body ], [ 0, %while.body.preheader ]
  %res.08 = phi i64 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %conv = zext i32 %i.09 to i64
  %rem = srem i64 %conv, 5
  %add = sub i64 %rem, %res.08
  %inc1 = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i64 %res.0.lcssa
}

; CHECK-LABEL: do_with_i64_udiv
; CHECK-NOT: llvm.set.loop.iterations
; CHECK-NOT: llvm.loop.decrement
define i64 @do_with_i64_udiv(i32 %n) {
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %i.09 = phi i32 [ %inc1, %while.body ], [ 0, %while.body.preheader ]
  %res.08 = phi i64 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %conv = zext i32 %i.09 to i64
  %rem = udiv i64 %conv, 5
  %add = add i64 %rem, %res.08
  %inc1 = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i64 %res.0.lcssa
}

; CHECK-LABEL: do_with_i64_sdiv
; CHECK-NOT: call void @llvm.set.loop.iterations
; CHECK-NOT: call i32 @llvm.loop.decrement
define i64 @do_with_i64_sdiv(i32 %n) {
entry:
  %cmp7 = icmp eq i32 %n, 0
  br i1 %cmp7, label %while.end, label %while.body.preheader

while.body.preheader:
  br label %while.body

while.body:
  %i.09 = phi i32 [ %inc1, %while.body ], [ 0, %while.body.preheader ]
  %res.08 = phi i64 [ %add, %while.body ], [ 0, %while.body.preheader ]
  %conv = zext i32 %i.09 to i64
  %rem = sdiv i64 %conv, 5
  %add = sub i64 %rem, %res.08
  %inc1 = add nuw i32 %i.09, 1
  %exitcond = icmp eq i32 %inc1, %n
  br i1 %exitcond, label %while.end.loopexit, label %while.body

while.end.loopexit:
  br label %while.end

while.end:
  %res.0.lcssa = phi i64 [ 0, %entry ], [ %add, %while.end.loopexit ]
  ret i64 %res.0.lcssa
}
