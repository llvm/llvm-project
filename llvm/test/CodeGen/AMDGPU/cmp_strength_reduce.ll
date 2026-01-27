define i32 @cmp_strength_reduce(i64 inreg %val0) {
  %result2 = lshr i64 %val0, 32
  %cmp = icmp ne i64 %result2, 42
  %zext = zext i1 %cmp to i32
  ret i32 %zext
}

define i32 @cmp_strength_reduce_low32(i64 inreg %val0) {
  %result2 = and i64 %val0, 4294967295
  %cmp = icmp ne i64 %result2, 42
  %zext = zext i1 %cmp to i32
  ret i32 %zext
}

define i32 @cmp_strength_reduce_low31(i64 inreg %val0) {
  %result2 = and i64 %val0, 2147483647
  %cmp = icmp ne i64 %result2, 42
  %zext = zext i1 %cmp to i32
  ret i32 %zext
}

define i32 @cmp_strength_reduce_high32(i64 inreg %val0) {
  %result2 = and i64 %val0, 18446744069414584320
  %cmp = icmp ne i64 %result2, 17883651179481661440
  %zext = zext i1 %cmp to i32
  ret i32 %zext
}

