define void @foo(ptr %pl, ptr %ps) {
  %gep_l0 = getelementptr inbounds i32, ptr %pl, i32 92
  %gep_l1 = getelementptr inbounds i32, ptr %pl, i32 0
  %gep_l2 = getelementptr inbounds i32, ptr %pl, i32 2
  %gep_l3 = getelementptr inbounds i32, ptr %pl, i32 3
  %gep_l4 = getelementptr inbounds i32, ptr %pl, i32 4
  %gep_l5 = getelementptr inbounds i32, ptr %pl, i32 5
  %gep_l6 = getelementptr inbounds i32, ptr %pl, i32 7
  %gep_l7 = getelementptr inbounds i32, ptr %pl, i32 93

  %load0  = load i32, ptr %gep_l0 , align 1
  %load1  = load i32, ptr %gep_l1 , align 1
  %load2  = load i32, ptr %gep_l2 , align 1
  %load3  = load i32, ptr %gep_l3 , align 1
  %load4  = load i32, ptr %gep_l4 , align 1
  %load5  = load i32, ptr %gep_l5 , align 1
  %load6  = load i32, ptr %gep_l6 , align 1
  %load7  = load i32, ptr %gep_l7 , align 1

  %add6 = add i32 %load6, 2
  %add7 = add i32 %load7, 2

  %gep_s0 = getelementptr inbounds i32, ptr %ps, i32 0
  %gep_s1 = getelementptr inbounds i32, ptr %ps, i32 1
  %gep_s2 = getelementptr inbounds i32, ptr %ps, i32 2
  %gep_s3 = getelementptr inbounds i32, ptr %ps, i32 3
  %gep_s4 = getelementptr inbounds i32, ptr %ps, i32 4
  %gep_s5 = getelementptr inbounds i32, ptr %ps, i32 5
  %gep_s6 = getelementptr inbounds i32, ptr %ps, i32 6
  %gep_s7 = getelementptr inbounds i32, ptr %ps, i32 7

  store i32 %load0, ptr %gep_s0, align 1
  store i32 %load1, ptr %gep_s1, align 1
  store i32 %load2, ptr %gep_s2, align 1
  store i32 %load3, ptr %gep_s3, align 1
  store i32 %load4, ptr %gep_s4, align 1
  store i32 %load5, ptr %gep_s5, align 1
  store i32 %add6, ptr %gep_s6, align 1
  store i32 %add7, ptr %gep_s7, align 1

  ret void
}