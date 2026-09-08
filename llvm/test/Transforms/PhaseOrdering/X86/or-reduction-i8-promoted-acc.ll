; RUN: opt -O3 -S < %s | FileCheck %s
; XFAIL: *

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; https://github.com/llvm/llvm-project/issues/222142
; The i8 accumulator is promoted to i32 by the frontend, so the loop-carried
; value is trunc i32 -> i8 on the backedge and zext i8 -> i32 on the next use.
; The whole OR chain should be narrowed back to i8 so that the loop is
; vectorized with <32 x i8> lanes rather than <8 x i32> lanes with a zext per
; load. Since the InstSimplify zext(trunc nuw X) -> X fold (#204089) the phi
; stays i32 and the vectorizer type shrinking does not recover the i8 width.
;
; unsigned char is_ascii(const unsigned char *data, unsigned long len) {
;   unsigned char res = 0;
;   for (unsigned long i = 0; i < len; ++i)
;     res |= data[i];
;   return res <= 0x7F;
; }

define zeroext i8 @is_ascii(ptr noundef %data, i64 noundef %len) #0 {
; CHECK-LABEL: @is_ascii(
; CHECK:       vector.body:
; CHECK:         phi <32 x i8>
; CHECK-NOT:     phi <8 x i32>
; CHECK:         load <32 x i8>
; CHECK-NOT:     zext <{{[0-9]+}} x i8>
; CHECK:         or <32 x i8>
; CHECK-NOT:     or <8 x i32>
; CHECK:       middle.block:
; CHECK:         call i8 @llvm.vector.reduce.or.v32i8(<32 x i8>
;
entry:
  %res = alloca i8, align 1
  %i = alloca i64, align 8
  store i8 0, ptr %res, align 1
  store i64 0, ptr %i, align 8
  br label %for.cond

for.cond:
  %i.val = load i64, ptr %i, align 8
  %cmp = icmp ult i64 %i.val, %len
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %idx = load i64, ptr %i, align 8
  %arrayidx = getelementptr inbounds nuw i8, ptr %data, i64 %idx
  %byte = load i8, ptr %arrayidx, align 1
  %byte.ext = zext i8 %byte to i32
  %res.val = load i8, ptr %res, align 1
  %res.ext = zext i8 %res.val to i32
  %or = or i32 %res.ext, %byte.ext
  %or.trunc = trunc i32 %or to i8
  store i8 %or.trunc, ptr %res, align 1
  %inc = add i64 %idx, 1
  store i64 %inc, ptr %i, align 8
  br label %for.cond

for.end:
  %res.final = load i8, ptr %res, align 1
  %res.final.ext = zext i8 %res.final to i32
  %cmp.ascii = icmp sle i32 %res.final.ext, 127
  %conv = zext i1 %cmp.ascii to i32
  %ret = trunc i32 %conv to i8
  ret i8 %ret
}

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" "target-features"="+avx2" }
