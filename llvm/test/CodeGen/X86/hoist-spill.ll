; RUN: llc < %s | FileCheck %s

; Check no spills to the same stack slot after hoisting.
; CHECK: mov{{.}} %{{.*}}, [[SPOFFSET1:-?[0-9]*]](%rsp)
; CHECK: mov{{.}} %{{.*}}, [[SPOFFSET2:-?[0-9]*]](%rsp)
; CHECK-NOT: mov{{.}} %{{.*}}, [[SPOFFSET1]](%rsp)
; CHECK-NOT: mov{{.}} %{{.*}}, [[SPOFFSET2]](%rsp)

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global i32*, align 8
@b = external global i32, align 4
@d = external global i32*, align 8

; Function Attrs: norecurse noreturn nounwind uwtable
define void @fn1(i32 %p1) {
entry:
  %tmp = load i32*, i32** @d, align 8
  %tmp1 = load i32*, i32** @a, align 8
  %tmp2 = sext i32 %p1 to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc14, %entry
  %indvar = phi i32 [ %indvar.next, %for.inc14 ], [ 0, %entry ]
  %indvars.iv30.in = phi i32 [ %indvars.iv30, %for.inc14 ], [ %p1, %entry ]
  %c.0 = phi i32 [ %inc15, %for.inc14 ], [ 1, %entry ]
  %k.0 = phi i32 [ %k.1.lcssa, %for.inc14 ], [ undef, %entry ]
  %tmp3 = icmp sgt i32 undef, 0
  %smax52 = select i1 %tmp3, i32 undef, i32 0
  %tmp4 = zext i32 %smax52 to i64
  %tmp5 = icmp sgt i64 undef, %tmp4
  %smax53 = select i1 %tmp5, i64 undef, i64 %tmp4
  %tmp6 = add nsw i64 %smax53, 1
  %tmp7 = sub nsw i64 %tmp6, %tmp4
  %tmp8 = add nsw i64 %tmp7, -8
  %tmp9 = sub i32 undef, %indvar
  %tmp10 = icmp sgt i64 %tmp2, 0
  %smax40 = select i1 %tmp10, i64 %tmp2, i64 0
  %scevgep41 = getelementptr i32, i32* %tmp1, i64 %smax40
  %indvars.iv30 = add i32 %indvars.iv30.in, -1
  %tmp11 = icmp sgt i32 %indvars.iv30, 0
  %smax = select i1 %tmp11, i32 %indvars.iv30, i32 0
  %tmp12 = zext i32 %smax to i64
  %sub = sub nsw i32 %p1, %c.0
  %cmp = icmp sgt i32 %sub, 0
  %sub. = select i1 %cmp, i32 %sub, i32 0
  %cmp326 = icmp sgt i32 %k.0, %p1
  br i1 %cmp326, label %for.cond4.preheader, label %for.body.preheader

for.cond4.preheader:                              ; preds = %for.body, %for.cond
  %k.1.lcssa = phi i32 [ %k.0, %for.cond ], [ %add, %for.body ]
  %cmp528 = icmp sgt i32 %sub., %p1
  br i1 %cmp528, label %for.inc14, label %for.body6.preheader

for.body6.preheader:                              ; preds = %for.cond4.preheader
  br i1 undef, label %for.body6, label %min.iters.checked

min.iters.checked:                                ; preds = %for.body6.preheader
  br i1 undef, label %for.body6, label %vector.memcheck

vector.memcheck:                                  ; preds = %min.iters.checked
  %bound1 = icmp ule i32* undef, %scevgep41
  %memcheck.conflict = and i1 undef, %bound1
  br i1 %memcheck.conflict, label %for.body6, label %vector.body.preheader

vector.body.preheader:                            ; preds = %vector.memcheck
  %lcmp.mod = icmp eq i64 undef, 0
  br i1 %lcmp.mod, label %vector.body.preheader.split, label %vector.body.prol

vector.body.prol:                                 ; preds = %vector.body.prol, %vector.body.preheader
  %prol.iter.cmp = icmp eq i64 undef, 0
  br i1 %prol.iter.cmp, label %vector.body.preheader.split, label %vector.body.prol

vector.body.preheader.split:                      ; preds = %vector.body.prol, %vector.body.preheader
  %tmp13 = icmp ult i64 %tmp8, 24
  br i1 %tmp13, label %middle.block, label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.body.preheader.split
  %index = phi i64 [ %index.next.3, %vector.body ], [ 0, %vector.body.preheader.split ]
  %index.next = add i64 %index, 8
  %offset.idx.1 = add i64 %tmp12, %index.next
  %tmp14 = getelementptr inbounds i32, i32* %tmp, i64 %offset.idx.1
  %tmp15 = bitcast i32* %tmp14 to <4 x i32>*
  %wide.load.1 = load <4 x i32>, <4 x i32>* %tmp15, align 4
  %tmp16 = getelementptr inbounds i32, i32* %tmp1, i64 %offset.idx.1
  %tmp17 = bitcast i32* %tmp16 to <4 x i32>*
  store <4 x i32> %wide.load.1, <4 x i32>* %tmp17, align 4
  %index.next.3 = add i64 %index, 32
  br i1 undef, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body, %vector.body.preheader.split
  br i1 undef, label %for.inc14, label %for.body6

for.body.preheader:                               ; preds = %for.cond
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.preheader
  %k.127 = phi i32 [ %k.0, %for.body.preheader ], [ %add, %for.body ]
  %add = add nsw i32 %k.127, 1
  %tmp18 = load i32, i32* undef, align 4
  store i32 %tmp18, i32* @b, align 4
  br i1 undef, label %for.body, label %for.cond4.preheader

for.body6:                                        ; preds = %for.body6, %middle.block, %vector.memcheck, %min.iters.checked, %for.body6.preheader
  %indvars.iv32 = phi i64 [ undef, %for.body6 ], [ %tmp12, %vector.memcheck ], [ %tmp12, %min.iters.checked ], [ %tmp12, %for.body6.preheader ], [ undef, %middle.block ]
  %arrayidx8 = getelementptr inbounds i32, i32* %tmp, i64 %indvars.iv32
  %tmp19 = load i32, i32* %arrayidx8, align 4
  %arrayidx10 = getelementptr inbounds i32, i32* %tmp1, i64 %indvars.iv32
  store i32 %tmp19, i32* %arrayidx10, align 4
  %cmp5 = icmp slt i64 %indvars.iv32, undef
  br i1 %cmp5, label %for.body6, label %for.inc14

for.inc14:                                        ; preds = %for.body6, %middle.block, %for.cond4.preheader
  %inc15 = add nuw nsw i32 %c.0, 1
  %indvar.next = add i32 %indvar, 1
  br label %for.cond
}
