; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: llvm.umul.with.overflow
define i32 @sterix(i32, i8, i64) {
entry:
  %conv = zext i32 %0 to i64
  %conv1 = sext i8 %1 to i32
  %mul = mul i32 %conv1, 1945964878
  %sh_prom = trunc i64 %2 to i32
  %shr = lshr i32 %mul, %sh_prom
  %conv2 = zext i32 %shr to i64
  %mul3 = mul nuw nsw i64 %conv, %conv2
  %conv6 = and i64 %mul3, 4294967295
  %tobool = icmp ne i64 %conv6, %mul3
  br i1 %tobool, label %lor.end, label %lor.rhs

lor.rhs:
  %and = and i64 %2, %mul3
  %conv4 = trunc i64 %and to i32
  %tobool7 = icmp ne i32 %conv4, 0
  %lnot = xor i1 %tobool7, true
  br label %lor.end

lor.end:
  %3 = phi i1 [ true, %entry ], [ %lnot, %lor.rhs ]
  %conv8 = zext i1 %3 to i32
  ret i32 %conv8
}

