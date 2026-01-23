define i32 @cmp_strength_reduce(i64 inreg %val0) {
  %result2 = lshr i64 %val0, 32
  %cmp = icmp ne i64 %result2, 42
  %zext = zext i1 %cmp to i32
  ret i32 %zext
}
