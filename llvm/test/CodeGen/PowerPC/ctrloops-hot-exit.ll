; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 < %s | FileCheck %s

; If there is an exit edge known to be frequently taken,
; we should not transform this loop.

; A loop having a hot exit edge (exit in false branch)
define signext i64 @func() {
; CHECK: @func
; CHECK-NOT: mtctr
; CHECK-NOT: bdnz

entry:
  %a = alloca [1000 x i32], align 4
  %0 = bitcast [1000 x i32]* %a to i8*
  br label %for.body

for.body:
  %i.013 = phi i64 [ 0, %entry ], [ %inc, %if.end ]
  %b.012 = phi i64 [ 0, %entry ], [ %xor, %if.end ]
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* %a, i64 0, i64 %i.013
  %1 = load i32, i32* %arrayidx, align 4
  %tobool = icmp eq i32 %1, 0
  br i1 %tobool, label %if.end, label %cleanup, !prof !1

if.end:
  %xor = xor i64 %i.013, %b.012
  %inc = add nuw nsw i64 %i.013, 1
  %cmp = icmp ult i64 %inc, 1000
  br i1 %cmp, label %for.body, label %cleanup

cleanup:
  %res = phi i64 [ %b.012, %for.body ], [ %xor, %if.end ]
  ret i64 %res
}

; A loop having a cold exit edge (exit in false branch)
define signext i64 @func2() {
; CHECK: @func2
; CHECK: mtctr
; CHECK: bdnz

entry:
  %a = alloca [1000 x i32], align 4
  %0 = bitcast [1000 x i32]* %a to i8*
  br label %for.body

for.body:
  %i.013 = phi i64 [ 0, %entry ], [ %inc, %if.end ]
  %b.012 = phi i64 [ 0, %entry ], [ %xor, %if.end ]
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* %a, i64 0, i64 %i.013
  %1 = load i32, i32* %arrayidx, align 4
  %tobool = icmp eq i32 %1, 0
  br i1 %tobool, label %if.end, label %cleanup, !prof !2

if.end:
  %xor = xor i64 %i.013, %b.012
  %inc = add nuw nsw i64 %i.013, 1
  %cmp = icmp ult i64 %inc, 1000
  br i1 %cmp, label %for.body, label %cleanup

cleanup:
  %res = phi i64 [ %b.012, %for.body ], [ %xor, %if.end ]
  ret i64 %res
}

; A loop having an exit edge without profile data  (exit in false branch)
define signext i64 @func3() {
; CHECK: @func3
; CHECK: mtctr
; CHECK: bdnz

entry:
  %a = alloca [1000 x i32], align 4
  %0 = bitcast [1000 x i32]* %a to i8*
  br label %for.body

for.body:
  %i.013 = phi i64 [ 0, %entry ], [ %inc, %if.end ]
  %b.012 = phi i64 [ 0, %entry ], [ %xor, %if.end ]
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* %a, i64 0, i64 %i.013
  %1 = load i32, i32* %arrayidx, align 4
  %tobool = icmp eq i32 %1, 0
  br i1 %tobool, label %if.end, label %cleanup

if.end:
  %xor = xor i64 %i.013, %b.012
  %inc = add nuw nsw i64 %i.013, 1
  %cmp = icmp ult i64 %inc, 1000
  br i1 %cmp, label %for.body, label %cleanup

cleanup:
  %res = phi i64 [ %b.012, %for.body ], [ %xor, %if.end ]
  ret i64 %res
}

; A loop having a hot exit edge (exit in true branch)
define signext i64 @func4() {
; CHECK: @func4
; CHECK-NOT: mtctr
; CHECK-NOT: bdnz

entry:
  %a = alloca [1000 x i32], align 4
  %0 = bitcast [1000 x i32]* %a to i8*
  br label %for.body

for.body:
  %i.013 = phi i64 [ 0, %entry ], [ %inc, %if.end ]
  %b.012 = phi i64 [ 0, %entry ], [ %xor, %if.end ]
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* %a, i64 0, i64 %i.013
  %1 = load i32, i32* %arrayidx, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %cleanup, label %if.end, !prof !2

if.end:
  %xor = xor i64 %i.013, %b.012
  %inc = add nuw nsw i64 %i.013, 1
  %cmp = icmp ult i64 %inc, 1000
  br i1 %cmp, label %for.body, label %cleanup

cleanup:
  %res = phi i64 [ %b.012, %for.body ], [ %xor, %if.end ]
  ret i64 %res
}

; A loop having a cold exit edge (exit in true branch)
define signext i64 @func5() {
; CHECK: @func5
; CHECK: mtctr
; CHECK: bdnz

entry:
  %a = alloca [1000 x i32], align 4
  %0 = bitcast [1000 x i32]* %a to i8*
  br label %for.body

for.body:
  %i.013 = phi i64 [ 0, %entry ], [ %inc, %if.end ]
  %b.012 = phi i64 [ 0, %entry ], [ %xor, %if.end ]
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* %a, i64 0, i64 %i.013
  %1 = load i32, i32* %arrayidx, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %cleanup, label %if.end, !prof !1

if.end:
  %xor = xor i64 %i.013, %b.012
  %inc = add nuw nsw i64 %i.013, 1
  %cmp = icmp ult i64 %inc, 1000
  br i1 %cmp, label %for.body, label %cleanup

cleanup:
  %res = phi i64 [ %b.012, %for.body ], [ %xor, %if.end ]
  ret i64 %res
}

; A loop having an exit edge without profile data  (exit in true branch)
define signext i64 @func6() {
; CHECK: @func6
; CHECK: mtctr
; CHECK: bdnz

entry:
  %a = alloca [1000 x i32], align 4
  %0 = bitcast [1000 x i32]* %a to i8*
  br label %for.body

for.body:
  %i.013 = phi i64 [ 0, %entry ], [ %inc, %if.end ]
  %b.012 = phi i64 [ 0, %entry ], [ %xor, %if.end ]
  %arrayidx = getelementptr inbounds [1000 x i32], [1000 x i32]* %a, i64 0, i64 %i.013
  %1 = load i32, i32* %arrayidx, align 4
  %tobool = icmp ne i32 %1, 0
  br i1 %tobool, label %cleanup, label %if.end

if.end:
  %xor = xor i64 %i.013, %b.012
  %inc = add nuw nsw i64 %i.013, 1
  %cmp = icmp ult i64 %inc, 1000
  br i1 %cmp, label %for.body, label %cleanup

cleanup:
  %res = phi i64 [ %b.012, %for.body ], [ %xor, %if.end ]
  ret i64 %res
}

!1 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}
