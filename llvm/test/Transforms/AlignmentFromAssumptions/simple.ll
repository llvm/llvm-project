target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
; RUN: opt < %s -passes=alignment-from-assumptions -S | FileCheck %s

define i32 @foo(ptr nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i32 32)]
  %0 = load i32, ptr %a, align 4
  ret i32 %0

; CHECK-LABEL: @foo
; CHECK: load i32, ptr {{[^,]+}}, align 32
; CHECK: ret i32
}

define i32 @foo2(ptr nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i32 32, i32 24)]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 -2
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0

; CHECK-LABEL: @foo2
; CHECK: load i32, ptr {{[^,]+}}, align 16
; CHECK: ret i32
}

define i32 @foo2a(ptr nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i32 32, i32 28)]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 1
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0

; CHECK-LABEL: @foo2a
; CHECK: load i32, ptr {{[^,]+}}, align 32
; CHECK: ret i32
}

; TODO: this can be 8-bytes aligned
define i32 @foo2b(ptr nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i32 32, i32 28)]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 -1
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0

; CHECK-LABEL: @foo2b
; CHECK: load i32, ptr {{[^,]+}}, align 4
; CHECK: ret i32
}

define i32 @goo(ptr nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i32 32, i32 0)]
  %0 = load i32, ptr %a, align 4
  ret i32 %0

; CHECK-LABEL: @goo
; CHECK: load i32, ptr {{[^,]+}}, align 32
; CHECK: ret i32
}

define i32 @hoo(ptr nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i64 32, i32 0)]
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %r.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %r.06
  %indvars.iv.next = add i64 %indvars.iv, 8
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, 2048
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

; CHECK-LABEL: @hoo
; CHECK: load i32, ptr %arrayidx, align 32
; CHECK: ret i32 %add.lcssa
}

; test D66575
; def hoo2(a, id, num):
;   for i0 in range(id*64, 4096, num*64):
;     for i1 in range(0, 4096, 32):
;       for i2 in range(0, 4096, 32):
;         load(a, i0+i1+i2+32)
define void @hoo2(ptr nocapture %a, i64 %id, i64 %num) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i8 32, i64 0)]
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
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %t3
  %x = load i32, ptr %arrayidx, align 4

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
; CHECK: load i32, ptr %arrayidx, align 32
; CHECK: ret void
}

define i32 @joo(ptr nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i8 32, i8 0)]
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 4, %entry ], [ %indvars.iv.next, %for.body ]
  %r.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %r.06
  %indvars.iv.next = add i64 %indvars.iv, 8
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, 2048
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

; CHECK-LABEL: @joo
; CHECK: load i32, ptr %arrayidx, align 16
; CHECK: ret i32 %add.lcssa
}

define i32 @koo(ptr nocapture %a) nounwind uwtable readonly {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %r.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i8 32, i8 0)]
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %r.06
  %indvars.iv.next = add i64 %indvars.iv, 4
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, 2048
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

; CHECK-LABEL: @koo
; CHECK: load i32, ptr %arrayidx, align 16
; CHECK: ret i32 %add.lcssa
}

define i32 @koo2(ptr nocapture %a) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i128 32, i128 0)]
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ -4, %entry ], [ %indvars.iv.next, %for.body ]
  %r.06 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %r.06
  %indvars.iv.next = add i64 %indvars.iv, 4
  %1 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %1, 2048
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %add.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %add.lcssa

; CHECK-LABEL: @koo2
; CHECK: load i32, ptr %arrayidx, align 16
; CHECK: ret i32 %add.lcssa
}

define i32 @moo(ptr nocapture %a) nounwind uwtable {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i16 32)]
  tail call void @llvm.memset.p0.i64(ptr align 4 %a, i8 0, i64 64, i1 false)
  ret i32 undef

; CHECK-LABEL: @moo
; CHECK: @llvm.memset.p0.i64(ptr align 32 %a, i8 0, i64 64, i1 false)
; CHECK: ret i32 undef
}

define i32 @moo2(ptr nocapture %a, ptr nocapture %b) nounwind uwtable {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %b, i32 128)]
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i16 32)]
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a, ptr align 4 %b, i64 64, i1 false)
  ret i32 undef

; CHECK-LABEL: @moo2
; CHECK: @llvm.memcpy.p0.p0.i64(ptr align 32 %a, ptr align 128 %b, i64 64, i1 false)
; CHECK: ret i32 undef
}

define i32 @moo3(ptr nocapture %a, ptr nocapture %b) nounwind uwtable {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i16 32), "align"(ptr %b, i32 128)]
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a, ptr align 4 %b, i64 64, i1 false)
  ret i32 undef

; CHECK-LABEL: @moo3
; CHECK: @llvm.memcpy.p0.p0.i64(ptr align 32 %a, ptr align 128 %b, i64 64, i1 false)
; CHECK: ret i32 undef
}


; Variable alignments appear to be legal, don't crash
define i32 @pr51680(ptr nocapture %a, i32 %align) nounwind uwtable readonly {
entry:
  tail call void @llvm.assume(i1 true) ["align"(ptr %a, i32 %align)]
  %0 = load i32, ptr %a, align 4
  ret i32 %0

; CHECK-LABEL: @pr51680
; CHECK: load i32, ptr {{[^,]+}}, align 4
; CHECK: ret i32
}

declare void @llvm.assume(i1) nounwind

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture, i64, i1) nounwind

