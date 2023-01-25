; RUN: llc < %s -march=hexagon

@si = common global i32 0, align 4
@sll = common global i64 0, align 8

define void @test_op_ignore() nounwind {
entry:
  %t00 = atomicrmw add ptr @si, i32 1 monotonic
  %t01 = atomicrmw add ptr @sll, i64 1 monotonic
  %t10 = atomicrmw sub ptr @si, i32 1 monotonic
  %t11 = atomicrmw sub ptr @sll, i64 1 monotonic
  %t20 = atomicrmw or ptr @si, i32 1 monotonic
  %t21 = atomicrmw or ptr @sll, i64 1 monotonic
  %t30 = atomicrmw xor ptr @si, i32 1 monotonic
  %t31 = atomicrmw xor ptr @sll, i64 1 monotonic
  %t40 = atomicrmw and ptr @si, i32 1 monotonic
  %t41 = atomicrmw and ptr @sll, i64 1 monotonic
  %t50 = atomicrmw nand ptr @si, i32 1 monotonic
  %t51 = atomicrmw nand ptr @sll, i64 1 monotonic
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_fetch_and_op() nounwind {
entry:
  %t00 = atomicrmw add ptr @si, i32 11 monotonic
  store i32 %t00, ptr @si, align 4
  %t01 = atomicrmw add ptr @sll, i64 11 monotonic
  store i64 %t01, ptr @sll, align 8
  %t10 = atomicrmw sub ptr @si, i32 11 monotonic
  store i32 %t10, ptr @si, align 4
  %t11 = atomicrmw sub ptr @sll, i64 11 monotonic
  store i64 %t11, ptr @sll, align 8
  %t20 = atomicrmw or ptr @si, i32 11 monotonic
  store i32 %t20, ptr @si, align 4
  %t21 = atomicrmw or ptr @sll, i64 11 monotonic
  store i64 %t21, ptr @sll, align 8
  %t30 = atomicrmw xor ptr @si, i32 11 monotonic
  store i32 %t30, ptr @si, align 4
  %t31 = atomicrmw xor ptr @sll, i64 11 monotonic
  store i64 %t31, ptr @sll, align 8
  %t40 = atomicrmw and ptr @si, i32 11 monotonic
  store i32 %t40, ptr @si, align 4
  %t41 = atomicrmw and ptr @sll, i64 11 monotonic
  store i64 %t41, ptr @sll, align 8
  %t50 = atomicrmw nand ptr @si, i32 11 monotonic
  store i32 %t50, ptr @si, align 4
  %t51 = atomicrmw nand ptr @sll, i64 11 monotonic
  store i64 %t51, ptr @sll, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}

define void @test_lock() nounwind {
entry:
  %t00 = atomicrmw xchg ptr @si, i32 1 monotonic
  store i32 %t00, ptr @si, align 4
  %t01 = atomicrmw xchg ptr @sll, i64 1 monotonic
  store i64 %t01, ptr @sll, align 8
  fence seq_cst
  store volatile i32 0, ptr @si, align 4
  store volatile i64 0, ptr @sll, align 8
  br label %return

return:                                           ; preds = %entry
  ret void
}


define i64 @fred() nounwind {
entry:
  %s0 = cmpxchg ptr undef, i32 undef, i32 undef seq_cst seq_cst
  %s1 = extractvalue { i32, i1 } %s0, 0
  %t0 = cmpxchg ptr undef, i64 undef, i64 undef seq_cst seq_cst
  %t1 = extractvalue { i64, i1 } %t0, 0
  %u0 = zext i32 %s1 to i64
  %u1 = add i64 %u0, %t1
  ret i64 %u1
}

