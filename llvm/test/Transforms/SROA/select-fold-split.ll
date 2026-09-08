; RUN: opt -passes='sroa<modify-cfg>' -S < %s | FileCheck %s

%struct.T = type { i32, i32 }

; CHECK-LABEL: @test_select_fold_split
; CHECK-NEXT: [[SEL:%.*]] = select i1 [[COND:%.*]], i32 1, i32 3
; CHECK-NEXT: ret i32 [[SEL]]
define i32 @test_select_fold_split(i1 %cond) {
  %alloc0 = alloca %struct.T, align 8
  %alloc1 = alloca %struct.T, align 8
  store %struct.T { i32 0, i32 1 }, ptr %alloc0
  store %struct.T { i32 2, i32 3 }, ptr %alloc1
  %sel = select i1 %cond, ptr %alloc0, ptr %alloc1
  %gep = getelementptr inbounds %struct.T, ptr %sel, i32 0, i32 1
  %val = load i32, ptr %gep
  ret i32 %val
}

; CHECK-LABEL: @test_select_fold_split_zero
; CHECK-NEXT: [[SEL:%.*]] = select i1 [[COND:%.*]], i32 0, i32 2
; CHECK-NEXT: ret i32 [[SEL]]
define i32 @test_select_fold_split_zero(i1 %cond) {
  %alloc0 = alloca %struct.T, align 8
  %alloc1 = alloca %struct.T, align 8
  store %struct.T { i32 0, i32 1 }, ptr %alloc0
  store %struct.T { i32 2, i32 3 }, ptr %alloc1
  %sel = select i1 %cond, ptr %alloc0, ptr %alloc1
  %gep = getelementptr inbounds %struct.T, ptr %sel, i32 0, i32 0
  %val = load i32, ptr %gep
  ret i32 %val
}

; CHECK-LABEL: @test_select_fold_split_var_gep
; CHECK-NEXT: [[ALLOC0:%.*]] = alloca %struct.T, align 8
; CHECK-NEXT: [[ALLOC1:%.*]] = alloca %struct.T, align 8
; CHECK-NEXT: [[ELEM0:%.*]] = getelementptr inbounds %struct.T, ptr [[ALLOC0]], i32 0, i32 0
; CHECK-NEXT: store i32 0, ptr [[ELEM0]], align 4
; CHECK-NEXT: [[ELEM1:%.*]] = getelementptr inbounds %struct.T, ptr [[ALLOC0]], i32 0, i32 1
; CHECK-NEXT: store i32 1, ptr [[ELEM1]], align 4
; CHECK-NEXT: [[ELEM2:%.*]] = getelementptr inbounds %struct.T, ptr [[ALLOC1]], i32 0, i32 0
; CHECK-NEXT: store i32 2, ptr [[ELEM2]], align 4
; CHECK-NEXT: [[ELEM3:%.*]] = getelementptr inbounds %struct.T, ptr [[ALLOC1]], i32 0, i32 1
; CHECK-NEXT: store i32 3, ptr [[ELEM3]], align 4
; CHECK-NEXT: [[SEL:%.*]] = select i1 [[COND:%.*]], ptr [[ALLOC0]], ptr [[ALLOC1]]
; CHECK-NEXT: [[GEP:%.*]] = getelementptr inbounds %struct.T, ptr [[SEL]], i32 [[V:%.*]], i32 1
; CHECK-NEXT: [[RET:%.*]] = load i32, ptr [[GEP]], align 4
; CHECK-NEXT: ret i32 [[RET]]
define i32 @test_select_fold_split_var_gep(i1 %cond, i32 %v) {
  %alloc0 = alloca %struct.T, align 8
  %alloc1 = alloca %struct.T, align 8
  store %struct.T { i32 0, i32 1 }, ptr %alloc0
  store %struct.T { i32 2, i32 3 }, ptr %alloc1
  %sel = select i1 %cond, ptr %alloc0, ptr %alloc1
  %gep = getelementptr inbounds %struct.T, ptr %sel, i32 %v, i32 1
  %val = load i32, ptr %gep
 ret i32 %val
}

; CHECK-LABEL: @test_select_fold_split_addrspace
; CHECK-NEXT: [[SEL:%.*]] = select i1 [[COND:%.*]], i32 1, i32 3
; CHECK-NEXT: ret i32 [[SEL]]
define i32 @test_select_fold_split_addrspace(i1 %cond) {
  %alloc0 = alloca %struct.T, align 16
  %alloc1 = alloca %struct.T, align 16
  store %struct.T { i32 0, i32 1 }, ptr %alloc0
  store %struct.T { i32 2, i32 3 }, ptr %alloc1
  %sel = select i1 %cond, ptr %alloc0, ptr %alloc1
  %cast = addrspacecast ptr %sel to ptr addrspace(5)
  %gep = getelementptr inbounds %struct.T, ptr addrspace(5) %cast, i32 0, i32 1
  %val = load i32, ptr addrspace(5) %gep
  ret i32 %val
}

; CHECK-LABEL: @test_select_fold_split_multiple_users_same
; CHECK-NEXT: [[SEL0:%.*]] = select i1 [[COND:%.*]], i32 0, i32 2
; CHECK-NEXT: [[SEL1:%.*]] = select i1 [[COND:%.*]], i32 0, i32 2
; CHECK-NEXT: [[ADD:%.*]] = add i32 [[SEL0]], [[SEL1]]
; CHECK-NEXT: ret i32 [[ADD]]
define i32 @test_select_fold_split_multiple_users_same(i1 %cond) {
  %alloc0 = alloca %struct.T, align 8
  %alloc1 = alloca %struct.T, align 8
  store %struct.T { i32 0, i32 1 }, ptr %alloc0
  store %struct.T { i32 2, i32 3 }, ptr %alloc1
  %sel = select i1 %cond, ptr %alloc0, ptr %alloc1
  %gep0 = getelementptr inbounds %struct.T, ptr %sel, i32 0, i32 0
  %gep1 = getelementptr inbounds %struct.T, ptr %sel, i32 0, i32 0
  %val0 = load i32, ptr %gep0
  %val1 = load i32, ptr %gep1
  %val = add i32 %val0, %val1
  ret i32 %val
}

; CHECK-LABEL: @test_select_fold_split_multiple_users_different
; CHECK-NEXT: [[SEL0:%.*]] = select i1 [[COND:%.*]], i32 0, i32 2
; CHECK-NEXT: [[SEL1:%.*]] = select i1 [[COND:%.*]], i32 1, i32 3
; CHECK-NEXT: [[ADD:%.*]] = add i32 [[SEL0]], [[SEL1]]
; CHECK-NEXT: ret i32 [[ADD]]
define i32 @test_select_fold_split_multiple_users_different(i1 %cond) {
  %alloc0 = alloca %struct.T, align 8
  %alloc1 = alloca %struct.T, align 8
  store %struct.T { i32 0, i32 1 }, ptr %alloc0
  store %struct.T { i32 2, i32 3 }, ptr %alloc1
  %sel = select i1 %cond, ptr %alloc0, ptr %alloc1
  %gep0 = getelementptr inbounds %struct.T, ptr %sel, i32 0, i32 0
  %gep1 = getelementptr inbounds %struct.T, ptr %sel, i32 0, i32 1
  %val0 = load i32, ptr %gep0
  %val1 = load i32, ptr %gep1
  %val = add i32 %val0, %val1
  ret i32 %val
}

; CHECK-LABEL: @test_select_fold_split_gep_chain
; CHECK-NEXT: [[SEL:%.*]] = select i1 [[COND:%.*]], i32 2, i32 5
; CHECK-NEXT: ret i32 [[SEL]]
define i32 @test_select_fold_split_gep_chain(i1 %cond) {
  %alloc0 = alloca [3 x i32], align 8
  %alloc1 = alloca [3 x i32], align 8
  store [3 x i32] [i32 0, i32 1, i32 2], ptr %alloc0
  store [3 x i32] [i32 3, i32 4, i32 5], ptr %alloc1
  %sel = select i1 %cond, ptr %alloc0, ptr %alloc1
  %gep0 = getelementptr i32, ptr %sel, i32 0
  %gep1 = getelementptr i32, ptr %gep0, i32 1
  %gep2 = getelementptr i32, ptr %gep1, i32 1
  %val = load i32, ptr %gep2
  ret i32 %val
}

; CHECK-LABEL: @test_select_fold_split_zero_gep
; CHECK-NEXT: [[SEL:%.*]] = select i1 [[COND:%.*]], i32 1, i32 3
; CHECK-NEXT: ret i32 [[SEL]]
define i32 @test_select_fold_split_zero_gep(i1 %cond) {
  %alloc0 = alloca %struct.T, align 8
  %alloc1 = alloca %struct.T, align 8
  store %struct.T { i32 0, i32 1 }, ptr %alloc0
  store %struct.T { i32 2, i32 3 }, ptr %alloc1
  %sel = select i1 %cond, ptr %alloc0, ptr %alloc1
  %gep0 = getelementptr %struct.T, ptr %sel, i32 0
  %gep1 = getelementptr inbounds %struct.T, ptr %gep0, i32 0, i32 1
  %val = load i32, ptr %gep1
  ret i32 %val
}



; Check for correct addrspacecast insertion for loads.
; CHECK-LABEL: @test_select_fold_split_volatile
; CHECK-NEXT: [[SROA_A0:%.*]] = alloca i32, align 8
; CHECK-NEXT: [[SROA_A1:%.*]] = alloca i32, align 4
; CHECK-NEXT: [[SROA_B0:%.*]] = alloca i32, align 8
; CHECK-NEXT: [[SROA_B1:%.*]] = alloca i32, align 4
; CHECK-NEXT: store i32 0, ptr [[SROA_A0]], align 8
; CHECK-NEXT: store i32 1, ptr [[SROA_A1]], align 4
; CHECK-NEXT: store i32 2, ptr [[SROA_B0]], align 8
; CHECK-NEXT: store i32 3, ptr [[SROA_B1]], align 4
; CHECK-NEXT: [[CAST_B0:%.*]] = addrspacecast ptr [[SROA_B0]] to ptr addrspace(5)
; CHECK-NEXT: [[CAST_A0:%.*]] = addrspacecast ptr [[SROA_A0]] to ptr addrspace(5)
; CHECK-NEXT: [[SEL0:%.*]] = select i1 [[COND:%.*]], ptr addrspace(5) [[CAST_A0]], ptr addrspace(5) [[CAST_B0]]
; CHECK-NEXT: [[CAST_B1:%.*]] = addrspacecast ptr [[SROA_B1]] to ptr addrspace(5)
; CHECK-NEXT: [[CAST_A1:%.*]] = addrspacecast ptr [[SROA_A1]] to ptr addrspace(5)
; CHECK-NEXT: [[SEL1:%.*]] = select i1 [[COND]], ptr addrspace(5) [[CAST_A1]], ptr addrspace(5) [[CAST_B1]]
; CHECK-NEXT: [[VAL1:%.*]] = load volatile i32, ptr addrspace(5) [[SEL0]], align 4
; CHECK-NEXT: [[VAL2:%.*]] = load volatile i32, ptr addrspace(5) [[SEL1]], align 4
; CHECK-NEXT: [[ADD:%.*]] = add i32 [[VAL1]], [[VAL2]]
; CHECK-NEXT: ret i32 [[ADD]]
define i32 @test_select_fold_split_volatile(i1 %cond) {
  %alloc0 = alloca %struct.T, align 8
  %alloc1 = alloca %struct.T, align 8
  store %struct.T { i32 0, i32 1 }, ptr %alloc0
  store %struct.T { i32 2, i32 3 }, ptr %alloc1
  %sel = select i1 %cond, ptr %alloc0, ptr %alloc1
  %cast = addrspacecast ptr %sel to ptr addrspace(5)
  %gep0 = getelementptr inbounds %struct.T, ptr addrspace(5) %cast, i32 0, i32 0
  %gep1 = getelementptr inbounds %struct.T, ptr addrspace(5) %cast, i32 0, i32 1
  %val1 = load volatile i32, ptr addrspace(5) %gep0
  %val2 = load volatile i32, ptr addrspace(5) %gep1
  %val3 = add i32 %val1, %val2
  ret i32 %val3
}
