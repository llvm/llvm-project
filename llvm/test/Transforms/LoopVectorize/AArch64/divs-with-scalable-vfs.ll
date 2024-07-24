; REQUIRES: asserts
; RUN: not --crash opt -p loop-vectorize -mtriple aarch64 -mcpu=neoverse-v1 -S %s

; Test case for https://github.com/llvm/llvm-project/issues/94328.
define void @sdiv_feeding_gep(ptr %dst, i32 %x, i64 %M, i64 %conv6, i64 %N) {
entry:
  %conv61 = zext i32 %x to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %div18 = sdiv i64 %M, %conv6
  %conv20 = trunc i64 %div18 to i32
  %mul30 = mul i64 %div18, %conv61
  %sub31 = sub i64 %iv, %mul30
  %conv34 = trunc i64 %sub31 to i32
  %mul35 = mul i32 %x, %conv20
  %add36 = add i32 %mul35, %conv34
  %idxprom = sext i32 %add36 to i64
  %gep = getelementptr double, ptr %dst, i64 %idxprom
  store double 0.000000e+00, ptr %gep, align 8
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %N
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define void @sdiv_feeding_gep_predicated(ptr %dst, i32 %x, i64 %M, i64 %conv6, i64 %N) {
entry:
  %conv61 = zext i32 %x to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop.latch ]
  %c = icmp ule i64 %iv, %M
  br i1 %c, label %then, label %loop.latch

then:
  %div18 = sdiv i64 %M, %conv6
  %conv20 = trunc i64 %div18 to i32
  %mul30 = mul i64 %div18, %conv61
  %sub31 = sub i64 %iv, %mul30
  %conv34 = trunc i64 %sub31 to i32
  %mul35 = mul i32 %x, %conv20
  %add36 = add i32 %mul35, %conv34
  %idxprom = sext i32 %add36 to i64
  %gep = getelementptr double, ptr %dst, i64 %idxprom
  store double 0.000000e+00, ptr %gep, align 8
  br label %loop.latch

loop.latch:
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %N
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

; Test case for https://github.com/llvm/llvm-project/issues/80416.
define void @udiv_urem_feeding_gep(i64 %x, ptr %dst, i64 %N) {
entry:
  %mul.1.i = mul i64 %x, %x
  %mul.2.i = mul i64 %mul.1.i, %x
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %div.i = udiv i64 %iv, %mul.2.i
  %rem.i = urem i64 %iv, %mul.2.i
  %div.1.i = udiv i64 %rem.i, %mul.1.i
  %rem.1.i = urem i64 %rem.i, %mul.1.i
  %div.2.i = udiv i64 %rem.1.i, %x
  %rem.2.i = urem i64 %rem.1.i, %x
  %mul.i = mul i64 %x, %div.i
  %add.i = add i64 %mul.i, %div.1.i
  %mul.1.i9 = mul i64 %add.i, %x
  %add.1.i = add i64 %mul.1.i9, %div.2.i
  %mul.2.i11 = mul i64 %add.1.i, %x
  %add.2.i = add i64 %mul.2.i11, %rem.2.i
  %sext.i = shl i64 %add.2.i, 32
  %conv6.i = ashr i64 %sext.i, 32
  %gep = getelementptr i64, ptr %dst, i64 %conv6.i
  store i64 %div.i, ptr %gep, align 4
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret void
}
