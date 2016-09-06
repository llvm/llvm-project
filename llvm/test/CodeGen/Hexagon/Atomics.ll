; RUN: llc < %s -march=hexagon

@si = common global i32 0, align 4
@sll = common global i64 0, align 8

define void @test_op_ignore() nounwind {
entry:
  %t00 = atomicrmw add i32* @si, i32 1 monotonic
  %t01 = atomicrmw add i64* @sll, i64 1 monotonic
  %t10 = atomicrmw sub i32* @si, i32 1 monotonic
  %t11 = atomicrmw sub i64* @sll, i64 1 monotonic
  %t20 = atomicrmw or i32* @si, i32 1 monotonic
  %t21 = atomicrmw or i64* @sll, i64 1 monotonic
  %t30 = atomicrmw xor i32* @si, i32 1 monotonic
  %t31 = atomicrmw xor i64* @sll, i64 1 monotonic
  %t40 = atomicrmw and i32* @si, i32 1 monotonic
  %t41 = atomicrmw and i64* @sll, i64 1 monotonic
  %t50 = atomicrmw nand i32* @si, i32 1 monotonic
  %t51 = atomicrmw nand i64* @sll, i64 1 monotonic
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_fetch_and_op() nounwind {
entry:
  %t00 = atomicrmw add i32* @si, i32 11 monotonic
  store i32 %t00, i32* @si, align 4
  %t01 = atomicrmw add i64* @sll, i64 11 monotonic
  store i64 %t01, i64* @sll, align 8
  %t10 = atomicrmw sub i32* @si, i32 11 monotonic
  store i32 %t10, i32* @si, align 4
  %t11 = atomicrmw sub i64* @sll, i64 11 monotonic
  store i64 %t11, i64* @sll, align 8
  %t20 = atomicrmw or i32* @si, i32 11 monotonic
  store i32 %t20, i32* @si, align 4
  %t21 = atomicrmw or i64* @sll, i64 11 monotonic
  store i64 %t21, i64* @sll, align 8
  %t30 = atomicrmw xor i32* @si, i32 11 monotonic
  store i32 %t30, i32* @si, align 4
  %t31 = atomicrmw xor i64* @sll, i64 11 monotonic
  store i64 %t31, i64* @sll, align 8
  %t40 = atomicrmw and i32* @si, i32 11 monotonic
  store i32 %t40, i32* @si, align 4
  %t41 = atomicrmw and i64* @sll, i64 11 monotonic
  store i64 %t41, i64* @sll, align 8
  %t50 = atomicrmw nand i32* @si, i32 11 monotonic
  store i32 %t50, i32* @si, align 4
  %t51 = atomicrmw nand i64* @sll, i64 11 monotonic
  store i64 %t51, i64* @sll, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_lock() nounwind {
entry:
  %t00 = atomicrmw xchg i32* @si, i32 1 monotonic
  store i32 %t00, i32* @si, align 4
  %t01 = atomicrmw xchg i64* @sll, i64 1 monotonic
  store i64 %t01, i64* @sll, align 8
  fence seq_cst
  store volatile i32 0, i32* @si, align 4
  store volatile i64 0, i64* @sll, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}


define i64 @fred() nounwind {
entry:
  %s0 = cmpxchg i32* undef, i32 undef, i32 undef seq_cst seq_cst
  %s1 = extractvalue { i32, i1 } %s0, 0
  %t0 = cmpxchg i64* undef, i64 undef, i64 undef seq_cst seq_cst
  %t1 = extractvalue { i64, i1 } %t0, 0
  %u0 = zext i32 %s1 to i64
  %u1 = add i64 %u0, %t1
  ret i64 %u1
}

