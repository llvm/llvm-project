; RUN: opt -loop-reduce -S < %s | FileCheck %s
; PR9939

; LSR should properly handle the post-inc offset when folding the
; non-IV operand of an icmp into the IV.

; CHECK:   [[r1:%[a-z0-9\.]+]] = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
; CHECK:   [[r2:%[a-z0-9\.]+]] = lshr exact i64 [[r1]], 1
; CHECK: for.body.lr.ph:
; CHECK:   [[r3:%[a-z0-9]+]] = shl i64 [[r2]], 1
; CHECK:   br label %for.body
; CHECK: for.body:
; CHECK:   %lsr.iv2 = phi i64 [ %lsr.iv.next, %for.body ], [ [[r3]], %for.body.lr.ph ]
; CHECK:   %lsr.iv.next = add i64 %lsr.iv2, -2
; CHECK:   %lsr.iv.next3 = inttoptr i64 %lsr.iv.next to i16*
; CHECK:   %cmp27 = icmp eq i16* %lsr.iv.next3, null

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

%struct.Vector2 = type { i16*, [64 x i16], i32 }

@.str = private unnamed_addr constant [37 x i8] c"0123456789abcdefghijklmnopqrstuvwxyz\00"

define void @_Z15IntegerToStringjjR7Vector2(i32 %i, i32 %radix, %struct.Vector2* nocapture %result) nounwind noinline {
entry:
  %buffer = alloca [33 x i16], align 16
  %add.ptr = getelementptr inbounds [33 x i16], [33 x i16]* %buffer, i64 0, i64 33
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %0 = phi i64 [ %indvar.next44, %do.body ], [ 0, %entry ]
  %i.addr.0 = phi i32 [ %div, %do.body ], [ %i, %entry ]
  %tmp51 = sub i64 32, %0
  %incdec.ptr = getelementptr [33 x i16], [33 x i16]* %buffer, i64 0, i64 %tmp51
  %rem = urem i32 %i.addr.0, 10
  %div = udiv i32 %i.addr.0, 10
  %idxprom = zext i32 %rem to i64
  %arrayidx = getelementptr inbounds [37 x i8], [37 x i8]* @.str, i64 0, i64 %idxprom
  %tmp5 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %tmp5 to i16
  store i16 %conv, i16* %incdec.ptr, align 2
  %1 = icmp ugt i32 %i.addr.0, 9
  %indvar.next44 = add i64 %0, 1
  br i1 %1, label %do.body, label %do.end

do.end:                                           ; preds = %do.body
  %xap.0 = inttoptr i64 %0 to i1*
  %cap.0 = ptrtoint i1* %xap.0 to i64
  %sub.ptr.lhs.cast = ptrtoint i16* %add.ptr to i64
  %sub.ptr.rhs.cast = ptrtoint i16* %incdec.ptr to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div39 = lshr exact i64 %sub.ptr.sub, 1
  %conv11 = trunc i64 %sub.ptr.div39 to i32
  %mLength = getelementptr inbounds %struct.Vector2, %struct.Vector2* %result, i64 0, i32 2
  %idx.ext21 = bitcast i64 %sub.ptr.div39 to i64
  %incdec.ptr.sum = add i64 %idx.ext21, -1
  %cp.0.sum = sub i64 %incdec.ptr.sum, %0
  %add.ptr22 = getelementptr [33 x i16], [33 x i16]* %buffer, i64 1, i64 %cp.0.sum
  %cmp2740 = icmp eq i64 %idx.ext21, 0
  br i1 %cmp2740, label %for.end, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %do.end
  %tmp16 = load i32, i32* %mLength, align 4
  %mBegin = getelementptr inbounds %struct.Vector2, %struct.Vector2* %result, i64 0, i32 0
  %tmp14 = load i16*, i16** %mBegin, align 8
  %tmp48 = zext i32 %tmp16 to i64
  br label %for.body

for.body:                                         ; preds = %for.body, %for.body.lr.ph
  %indvar = phi i64 [ 0, %for.body.lr.ph ], [ %indvar.next, %for.body ]
  %tmp46 = add i64 %tmp51, %indvar
  %p.042 = getelementptr [33 x i16], [33 x i16]* %buffer, i64 0, i64 %tmp46
  %tmp47 = sub i64 %indvar, %0
  %incdec.ptr32 = getelementptr [33 x i16], [33 x i16]* %buffer, i64 1, i64 %tmp47
  %tmp49 = add i64 %tmp48, %indvar
  %dst.041 = getelementptr i16, i16* %tmp14, i64 %tmp49
  %tmp29 = load i16, i16* %p.042, align 2
  store i16 %tmp29, i16* %dst.041, align 2
  %cmp27 = icmp eq i16* %incdec.ptr32, %add.ptr22
  %indvar.next = add i64 %indvar, 1
  br i1 %cmp27, label %for.end.loopexit, label %for.body

for.end.loopexit:                                 ; preds = %for.body
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %do.end
  %tmp38 = load i32, i32* %mLength, align 4
  %add = add i32 %tmp38, %conv11
  store i32 %add, i32* %mLength, align 4
  ret void
}
