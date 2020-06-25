target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
; RUN: opt < %s -alignment-from-assumptions -S | FileCheck %s
; RUN: opt < %s -passes=alignment-from-assumptions -S | FileCheck %s

define i32 @foo(i32* nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(i32* %a, i32 32)]
  %0 = load i32, i32* %a, align 4
  ret i32 %0

; CHECK-LABEL: @foo
; CHECK: load i32, i32* {{[^,]+}}, align 32
; CHECK: ret i32
}

define i32 @foo2(i32* nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(i32* %a, i32 32, i32 24)]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 2
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0

; CHECK-LABEL: @foo2
; CHECK: load i32, i32* {{[^,]+}}, align 16
; CHECK: ret i32
}

define i32 @foo2a(i32* nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(i32* %a, i32 32, i32 28)]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 -1
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0

; CHECK-LABEL: @foo2a
; CHECK: load i32, i32* {{[^,]+}}, align 32
; CHECK: ret i32
}

define i32 @goo(i32* nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(i32* %a, i32 32, i32 0)]
  %0 = load i32, i32* %a, align 4
  ret i32 %0

; CHECK-LABEL: @goo
; CHECK: load i32, i32* {{[^,]+}}, align 32
; CHECK: ret i32
}

define i32 @hoo(i32* nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(i32* %a, i64 32, i32 0)]
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

; test D66575
; def hoo2(a, id, num):
;   for i0 in range(id*64, 4096, num*64):
;     for i1 in range(0, 4096, 32):
;       for i2 in range(0, 4096, 32):
;         load(a, i0+i1+i2+32)
define void @hoo2(i32* nocapture %a, i64 %id, i64 %num) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(i32* %a, i8 32, i64 0)]
  %id.mul = shl nsw i64 %id, 6
  %num.mul = shl nsw i64 %num, 6
  br label %for0.body

for0.body:
  %i0 = phi i64 [ %id.mul, %entry ], [ %i0.next, %for0.end ]
  br label %for1.body

for1.body:
  %i1 = phi i64 [ 0, %for0.body ], [ %i1.next, %for1.end ]
  br label %for2.body

for2.body:
  %i2 = phi i64 [ 0, %for1.body ], [ %i2.next, %for2.body ]

  %t1 = add nuw nsw i64 %i0, %i1
  %t2 = add nuw nsw i64 %t1, %i2
  %t3 = add nuw nsw i64 %t2, 32
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %t3
  %x = load i32, i32* %arrayidx, align 4

  %i2.next = add nuw nsw i64 %i2, 32
  %cmp2 = icmp ult i64 %i2.next, 4096
  br i1 %cmp2, label %for2.body, label %for1.end

for1.end:
  %i1.next = add nuw nsw i64 %i1, 32
  %cmp1 = icmp ult i64 %i1.next, 4096
  br i1 %cmp1, label %for1.body, label %for0.end

for0.end:
  %i0.next = add nuw nsw i64 %i0, %num.mul
  %cmp0 = icmp ult i64 %i0.next, 4096
  br i1 %cmp0, label %for0.body, label %return

return:
  ret void

; CHECK-LABEL: @hoo2
; CHECK: load i32, i32* %arrayidx, align 32
; CHECK: ret void
}

define i32 @joo(i32* nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(i32* %a, i8 32, i8 0)]
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
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %r.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  tail call void @llvm.assume(i1 true) ["align"(i32* %a, i8 32, i8 0)]
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
  tail call void @llvm.assume(i1 true) ["align"(i32* %a, i128 32, i128 0)]
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
  tail call void @llvm.assume(i1 true) ["align"(i32* %a, i16 32)]
  %0 = bitcast i32* %a to i8*
  tail call void @llvm.memset.p0i8.i64(i8* align 4 %0, i8 0, i64 64, i1 false)
  ret i32 undef

; CHECK-LABEL: @moo
; CHECK: @llvm.memset.p0i8.i64(i8* align 32 %0, i8 0, i64 64, i1 false)
; CHECK: ret i32 undef
}

define i32 @moo2(i32* nocapture %a, i32* nocapture %b) nounwind uwtable {
entry:
  tail call void @llvm.assume(i1 true) ["align"(i32* %b, i32 128)]
  %0 = bitcast i32* %a to i8*
  tail call void @llvm.assume(i1 true) ["align"(i8* %0, i16 32)]
  %1 = bitcast i32* %b to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %0, i8* align 4 %1, i64 64, i1 false)
  ret i32 undef

; CHECK-LABEL: @moo2
; CHECK: @llvm.memcpy.p0i8.p0i8.i64(i8* align 32 %0, i8* align 128 %1, i64 64, i1 false)
; CHECK: ret i32 undef
}

declare void @llvm.assume(i1) nounwind

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

