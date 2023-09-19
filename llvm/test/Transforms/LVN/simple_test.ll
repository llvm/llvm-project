define dso_local i32 @f(i32 noundef %a, i32 noundef %b, i32 noundef %c) {
entry:
  %add = add i32 %a, %b
  %mul = mul i32 %b, %c
  %add1 = add i32 %a, %b
  %mul2 = mul i32 %add, %mul
  %add3 = add i32 %mul2, %add1
  ret i32 %add3
}
