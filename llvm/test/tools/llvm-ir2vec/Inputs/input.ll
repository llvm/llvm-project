define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}

define i32 @multiply(i32 %x, i32 %y) {
entry:
  %prod = mul i32 %x, %y
  ret i32 %prod
}

define i32 @conditional(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %positive, label %negative

positive:
  %pos_val = add i32 %n, 10
  br label %exit

negative:
  %neg_val = sub i32 %n, 10
  br label %exit

exit:
  %result = phi i32 [ %pos_val, %positive ], [ %neg_val, %negative ]
  ret i32 %result
}
