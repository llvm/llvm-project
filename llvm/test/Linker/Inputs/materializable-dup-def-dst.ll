define i32 @mat_dup_test_1() {
entry:
  ret i32 42
}

define void @mat_dup_test_2(i32 %a, i32 %b) {
entry:
  %sum = add nsw i32 %a, %b
  %prod = mul nsw i32 %sum, 3
  ret void
}