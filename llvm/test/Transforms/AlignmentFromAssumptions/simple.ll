target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
; RUN: opt < %s -alignment-from-assumptions -S | FileCheck %s

define i32 @foo(i32* nocapture %a) nounwind uwtable readonly {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  %0 = load i32, i32* %a, align 4
  ret i32 %0

; CHECK-LABEL: @foo
; CHECK: load i32, i32* {{[^,]+}}, align 32
; CHECK: ret i32
}

define i32 @foo2(i32* nocapture %a) nounwind uwtable readonly {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %offsetptr = add i64 %ptrint, 24
  %maskedptr = and i64 %offsetptr, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 2
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0

; CHECK-LABEL: @foo2
; CHECK: load i32, i32* {{[^,]+}}, align 16
; CHECK: ret i32
}

define i32 @foo2a(i32* nocapture %a) nounwind uwtable readonly {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %offsetptr = add i64 %ptrint, 28
  %maskedptr = and i64 %offsetptr, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 -1
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0

; CHECK-LABEL: @foo2a
; CHECK: load i32, i32* {{[^,]+}}, align 32
; CHECK: ret i32
}

define i32 @goo(i32* nocapture %a) nounwind uwtable readonly {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  %0 = load i32, i32* %a, align 4
  ret i32 %0

; CHECK-LABEL: @goo
; CHECK: load i32, i32* {{[^,]+}}, align 32
; CHECK: ret i32
}

define i32 @hoo(i32* nocapture %a) nounwind uwtable readonly {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %r.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %r.06
  %indvars.iv.next = add i64 %indvars.iv, 8
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, 2048
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

; CHECK-LABEL: @hoo
; CHECK: load i32, i32* %arrayidx, align 32
; CHECK: ret i32 %add.lcssa
}

define i32 @joo(i32* nocapture %a) nounwind uwtable readonly {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 4, %entry ], [ %indvars.iv.next, %for.body ]
  %r.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %r.06
  %indvars.iv.next = add i64 %indvars.iv, 8
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, 2048
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

; CHECK-LABEL: @joo
; CHECK: load i32, i32* %arrayidx, align 16
; CHECK: ret i32 %add.lcssa
}

define i32 @koo(i32* nocapture %a) nounwind uwtable readonly {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %r.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %r.06
  %indvars.iv.next = add i64 %indvars.iv, 4
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, 2048
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

; CHECK-LABEL: @koo
; CHECK: load i32, i32* %arrayidx, align 16
; CHECK: ret i32 %add.lcssa
}

define i32 @koo2(i32* nocapture %a) nounwind uwtable readonly {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ -4, %entry ], [ %indvars.iv.next, %for.body ]
  %r.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %r.06
  %indvars.iv.next = add i64 %indvars.iv, 4
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, 2048
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

; CHECK-LABEL: @koo2
; CHECK: load i32, i32* %arrayidx, align 16
; CHECK: ret i32 %add.lcssa
}

define i32 @moo(i32* nocapture %a) nounwind uwtable {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  %0 = bitcast i32* %a to i8*
  tail call void @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 64, i32 4, i1 false)
  ret i32 undef

; CHECK-LABEL: @moo
; CHECK: @llvm.memset.p0i8.i64(i8* %0, i8 0, i64 64, i32 32, i1 false)
; CHECK: ret i32 undef
}

define i32 @moo2(i32* nocapture %a, i32* nocapture %b) nounwind uwtable {
entry:
  %ptrint = ptrtoint i32* %a to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  tail call void @llvm.assume(i1 %maskcond)
  %ptrint1 = ptrtoint i32* %b to i64
  %maskedptr3 = and i64 %ptrint1, 127
  %maskcond4 = icmp eq i64 %maskedptr3, 0
  tail call void @llvm.assume(i1 %maskcond4)
  %0 = bitcast i32* %a to i8*
  %1 = bitcast i32* %b to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %1, i64 64, i32 4, i1 false)
  ret i32 undef

; CHECK-LABEL: @moo2
; CHECK: @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %1, i64 64, i32 32, i1 false)
; CHECK: ret i32 undef
}

declare void @llvm.assume(i1) nounwind

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind

