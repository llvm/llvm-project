define i32 @heavy_pressure(i32 %arg) {
entry:
  ; 創建 32 個變數並強制它們同時存活
  %v0 = mul i32 %arg, 2
  %v1 = mul i32 %arg, 3
  %v2 = mul i32 %arg, 5
  %v3 = mul i32 %arg, 7
  %v4 = mul i32 %arg, 11
  %v5 = mul i32 %arg, 13
  %v6 = mul i32 %arg, 17
  %v7 = mul i32 %arg, 19
  %v8 = mul i32 %arg, 23
  %v9 = mul i32 %arg, 29
  %v10 = mul i32 %arg, 31
  %v11 = mul i32 %arg, 37
  %v12 = mul i32 %arg, 41
  %v13 = mul i32 %arg, 43
  %v14 = mul i32 %arg, 47
  %v15 = mul i32 %arg, 53
  
  ; 使用所有變數
  %s0 = add i32 %v0, %v1
  %s1 = add i32 %s0, %v2
  %s2 = add i32 %s1, %v3
  %s3 = add i32 %s2, %v4
  %s4 = add i32 %s3, %v5
  %s5 = add i32 %s4, %v6
  %s6 = add i32 %s5, %v7
  %s7 = add i32 %s6, %v8
  %s8 = add i32 %s7, %v9
  %s9 = add i32 %s8, %v10
  %s10 = add i32 %s9, %v11
  %s11 = add i32 %s10, %v12
  %s12 = add i32 %s11, %v13
  %s13 = add i32 %s12, %v14
  %s14 = add i32 %s13, %v15
  
  ; 再次使用所有原始變數強制延長生命週期
  %t0 = mul i32 %v0, %v15
  %t1 = mul i32 %v1, %v14
  %t2 = mul i32 %v2, %v13
  %t3 = mul i32 %v3, %v12
  %t4 = mul i32 %v4, %v11
  %t5 = mul i32 %v5, %v10
  %t6 = mul i32 %v6, %v9
  %t7 = mul i32 %v7, %v8
  
  %final = add i32 %s14, %t0
  %final1 = add i32 %final, %t1
  %final2 = add i32 %final1, %t2
  %final3 = add i32 %final2, %t3
  %final4 = add i32 %final3, %t4
  %final5 = add i32 %final4, %t5
  %final6 = add i32 %final5, %t6
  %final7 = add i32 %final6, %t7
  
  ret i32 %final7
}
