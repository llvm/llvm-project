define i32 @high_pressure() {
entry:
  %a1 = add i32 1, 2
  %a2 = add i32 %a1, 3
  %a3 = add i32 %a2, 4
  %a4 = add i32 %a3, 5
  %a5 = add i32 %a4, 6
  %a6 = add i32 %a5, 7
  %a7 = add i32 %a6, 8
  %a8 = add i32 %a7, 9
  %a9 = add i32 %a8, 10
  %a10 = add i32 %a9, 11
  %a11 = add i32 %a10, 12
  %a12 = add i32 %a11, 13
  %a13 = add i32 %a12, 14
  %a14 = add i32 %a13, 15
  %a15 = add i32 %a14, 16
  
  ; 強制所有變數同時存活
  %sum1 = add i32 %a1, %a2
  %sum2 = add i32 %sum1, %a3
  %sum3 = add i32 %sum2, %a4
  %sum4 = add i32 %sum3, %a5
  %sum5 = add i32 %sum4, %a6
  %sum6 = add i32 %sum5, %a7
  %sum7 = add i32 %sum6, %a8
  %sum8 = add i32 %sum7, %a9
  %sum9 = add i32 %sum8, %a10
  %sum10 = add i32 %sum9, %a11
  %sum11 = add i32 %sum10, %a12
  %sum12 = add i32 %sum11, %a13
  %sum13 = add i32 %sum12, %a14
  %sum14 = add i32 %sum13, %a15
  
  ret i32 %sum14
}
