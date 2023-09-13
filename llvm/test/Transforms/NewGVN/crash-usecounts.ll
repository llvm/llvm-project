; RUN: opt -passes=newgvn -disable-output < %s

define void @test(i1 %arg, i1 %arg1) {
bb:
  %alloca = alloca i64, i32 0, align 8
  %alloca2 = alloca i64, i32 0, align 8
  %load = load i64, ptr null, align 8
  %icmp = icmp ult i64 0, %load
  br i1 %icmp, label %bb24, label %bb25

bb24:                                             ; preds = %bb
  unreachable

bb25:                                             ; preds = %bb
  store i64 %load, ptr %alloca, align 8
  %load26 = load i64, ptr addrspace(1) null, align 8
  %icmp27 = icmp ugt i64 %load26, 1
  br i1 %icmp27, label %bb28, label %bb29

bb28:                                             ; preds = %bb25
  unreachable

bb29:                                             ; preds = %bb25
  %icmp30 = icmp ugt i64 %load26, 0
  br i1 %icmp30, label %bb34, label %bb31

bb31:                                             ; preds = %bb38, %bb29
  %load32 = load i64, ptr addrspace(1) null, align 8
  %icmp33 = icmp ugt i64 %load32, 1
  br i1 %icmp33, label %bb42, label %bb43

bb34:                                             ; preds = %bb29
  %load35 = load i64, ptr addrspace(1) null, align 8
  store i64 %load35, ptr null, align 8
  %load36 = load i64, ptr null, align 8
  br label %bb37

bb37:                                             ; preds = %bb37, %bb34
  %phi = phi i64 [ 0, %bb37 ], [ %load36, %bb34 ]
  %inttoptr = inttoptr i64 %phi to ptr addrspace(1)
  br i1 false, label %bb37, label %bb38

bb38:                                             ; preds = %bb37
  %load39 = load i64, ptr null, align 8
  %inttoptr40 = inttoptr i64 %load39 to ptr addrspace(1)
  %load41 = load i64, ptr addrspace(1) %inttoptr40, align 8
  %shl = shl i64 %load26, 0
  %and = and i64 %shl, 0
  br label %bb31

bb42:                                             ; preds = %bb31
  unreachable

bb43:                                             ; preds = %bb31
  %load44 = load i64, ptr null, align 8
  %load45 = load i64, ptr null, align 8
  %icmp46 = icmp ugt i64 1, %load45
  br i1 %icmp46, label %bb51, label %bb60

bb47:                                             ; preds = %bb60, %bb55, %bb51
  %load48 = load i64, ptr %alloca, align 8
  %inttoptr49 = inttoptr i64 %load48 to ptr addrspace(1)
  %load50 = load i64, ptr addrspace(1) %inttoptr49, align 8
  br i1 %arg, label %bb63, label %bb75

bb51:                                             ; preds = %bb43
  %load52 = load i64, ptr addrspace(1) null, align 8
  store i64 %load52, ptr %alloca2, align 8
  %load53 = load i64, ptr null, align 8
  store i64 %load53, ptr null, align 8
  %icmp54 = icmp ult i64 0, %load32
  br i1 %icmp54, label %bb55, label %bb47

bb55:                                             ; preds = %bb51
  %load56 = load i64, ptr null, align 8
  %or = or i64 %load56, 0
  %inttoptr57 = inttoptr i64 %or to ptr addrspace(1)
  %load58 = load i64, ptr addrspace(1) %inttoptr57, align 8
  %and59 = and i64 %load32, 0
  br label %bb47

bb60:                                             ; preds = %bb43
  %or61 = or i64 %load44, 0
  %load62 = load i64, ptr addrspace(1) null, align 8
  store i64 %load62, ptr null, align 8
  br label %bb47

bb63:                                             ; preds = %bb47
  %load64 = load i64, ptr addrspace(1) inttoptr (i64 64 to ptr addrspace(1)), align 8
  %inttoptr65 = inttoptr i64 %load64 to ptr addrspace(1)
  store i64 0, ptr addrspace(1) %inttoptr65, align 8
  %or66 = or i64 %load64, 4
  %inttoptr67 = inttoptr i64 %or66 to ptr addrspace(1)
  store i64 0, ptr addrspace(1) %inttoptr67, align 8
  %or68 = or i64 %load64, 36
  %inttoptr69 = inttoptr i64 %or68 to ptr addrspace(1)
  store i64 0, ptr addrspace(1) %inttoptr69, align 8
  %or70 = or i64 %load64, 68
  %inttoptr71 = inttoptr i64 %or70 to ptr addrspace(1)
  store i64 0, ptr addrspace(1) %inttoptr71, align 8
  %load72 = load i64, ptr null, align 8
  %shl73 = shl i64 %load72, 0
  %or74 = or i64 %shl73, 0
  store i64 %or74, ptr null, align 8
  unreachable

bb75:                                             ; preds = %bb47
  br i1 %arg1, label %bb88, label %bb76

bb76:                                             ; preds = %bb75
  %load77 = load i64, ptr addrspace(1) inttoptr (i64 64 to ptr addrspace(1)), align 8
  %inttoptr78 = inttoptr i64 %load77 to ptr addrspace(1)
  store i64 0, ptr addrspace(1) %inttoptr78, align 8
  %or79 = or i64 %load77, 4
  %inttoptr80 = inttoptr i64 %or79 to ptr addrspace(1)
  store i64 0, ptr addrspace(1) %inttoptr80, align 8
  %or81 = or i64 %load77, 36
  %inttoptr82 = inttoptr i64 %or81 to ptr addrspace(1)
  store i64 0, ptr addrspace(1) %inttoptr82, align 8
  %or83 = or i64 %load77, 68
  %inttoptr84 = inttoptr i64 %or83 to ptr addrspace(1)
  store i64 0, ptr addrspace(1) %inttoptr84, align 8
  %load85 = load i64, ptr null, align 8
  %shl86 = shl i64 %load85, 0
  %or87 = or i64 %shl86, 0
  store i64 %or87, ptr null, align 8
  unreachable

bb88:                                             ; preds = %bb75
  %load89 = load i64, ptr addrspace(1) null, align 8
  %icmp90 = icmp ugt i64 %load89, 0
  br i1 %icmp90, label %bb91, label %bb92

bb91:                                             ; preds = %bb88
  unreachable

bb92:                                             ; preds = %bb88
  br i1 false, label %bb93, label %bb95

bb93:                                             ; preds = %bb93, %bb92
  %phi94 = phi i64 [ 0, %bb93 ], [ %load89, %bb92 ]
  br label %bb93

bb95:                                             ; preds = %bb92
  %load96 = load i64, ptr null, align 8
  br label %bb98

bb97:                                             ; preds = %bb98
  ret void

bb98:                                             ; preds = %bb98, %bb95
  %phi99 = phi i64 [ %load96, %bb95 ], [ 0, %bb98 ]
  %inttoptr100 = inttoptr i64 %phi99 to ptr
  %load101 = load i64, ptr %inttoptr100, align 8
  br i1 false, label %bb98, label %bb97
}
