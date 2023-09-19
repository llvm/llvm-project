; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK:     OpEntryPoint Kernel %[[#f1:]] "writer"
; CHECK:     OpEntryPoint Kernel %[[#f2:]] "reader"
; CHECK-DAG: OpName %[[#a_var:]] "a_var"
; CHECK-DAG: OpName %[[#p_var:]] "p_var"
; CHECK-DAG: %[[#uchar:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#pt1:]] = OpTypePointer CrossWorkgroup %[[#uchar]]
; CHECK-DAG: %[[#arr2:]] = OpTypeArray %[[#uchar]]
; CHECK-DAG: %[[#pt2:]] = OpTypePointer CrossWorkgroup %[[#arr2]]
; CHECK-DAG: %[[#pt3:]] = OpTypePointer CrossWorkgroup %[[#pt1]]
; CHECK-DAG: %[[#a_var]] = OpVariable %[[#pt2]] CrossWorkgroup
; CHECK-DAG: %[[#const:]] = OpSpecConstantOp %[[#pt1]] 70 %[[#a_var]]
; CHECK-DAG: %[[#p_var]] = OpVariable %[[#pt3]] CrossWorkgroup %[[#const]]
@var = addrspace(1) global i8 0, align 1
@g_var = addrspace(1) global i8 1, align 1
@a_var = addrspace(1) global [2 x i8] c"\01\01", align 1
@p_var = addrspace(1) global i8 addrspace(1)* getelementptr inbounds ([2 x i8], [2 x i8] addrspace(1)* @a_var, i32 0, i64 1), align 8

define spir_func zeroext i8 @from_buf(i8 zeroext %a) {
entry:
  %tobool = icmp ne i8 %a, 0
  %i1promo = zext i1 %tobool to i8
  ret i8 %i1promo
}

define spir_func zeroext i8 @to_buf(i8 zeroext %a) {
entry:
  %i1trunc = trunc i8 %a to i1
  %frombool = select i1 %i1trunc, i8 1, i8 0
  %0 = and i8 %frombool, 1
  %tobool = icmp ne i8 %0, 0
  %conv = select i1 %tobool, i8 1, i8 0
  ret i8 %conv
}

define spir_kernel void @writer(i8 addrspace(1)* %src, i32 %idx) {
entry:
  %arrayidx = getelementptr inbounds i8, i8 addrspace(1)* %src, i64 0
  %0 = load i8, i8 addrspace(1)* %arrayidx, align 1
  %call = call spir_func zeroext i8 @from_buf(i8 zeroext %0)
  %i1trunc = trunc i8 %call to i1
  %frombool = select i1 %i1trunc, i8 1, i8 0
  store i8 %frombool, i8 addrspace(1)* @var, align 1
  %arrayidx1 = getelementptr inbounds i8, i8 addrspace(1)* %src, i64 1
  %1 = load i8, i8 addrspace(1)* %arrayidx1, align 1
  %call2 = call spir_func zeroext i8 @from_buf(i8 zeroext %1)
  %i1trunc1 = trunc i8 %call2 to i1
  %frombool3 = select i1 %i1trunc1, i8 1, i8 0
  store i8 %frombool3, i8 addrspace(1)* @g_var, align 1
  %arrayidx4 = getelementptr inbounds i8, i8 addrspace(1)* %src, i64 2
  %2 = load i8, i8 addrspace(1)* %arrayidx4, align 1
  %call5 = call spir_func zeroext i8 @from_buf(i8 zeroext %2)
  %i1trunc2 = trunc i8 %call5 to i1
  %frombool6 = select i1 %i1trunc2, i8 1, i8 0
  %3 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(1)* @a_var, i64 0, i64 0
  store i8 %frombool6, i8 addrspace(1)* %3, align 1
  %arrayidx7 = getelementptr inbounds i8, i8 addrspace(1)* %src, i64 3
  %4 = load i8, i8 addrspace(1)* %arrayidx7, align 1
  %call8 = call spir_func zeroext i8 @from_buf(i8 zeroext %4)
  %i1trunc3 = trunc i8 %call8 to i1
  %frombool9 = select i1 %i1trunc3, i8 1, i8 0
  %5 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(1)* @a_var, i64 0, i64 1
  store i8 %frombool9, i8 addrspace(1)* %5, align 1
  %idx.ext = zext i32 %idx to i64
  %add.ptr = getelementptr inbounds i8, i8 addrspace(1)* %3, i64 %idx.ext
  store i8 addrspace(1)* %add.ptr, i8 addrspace(1)* addrspace(1)* @p_var, align 8
  ret void
}

define spir_kernel void @reader(i8 addrspace(1)* %dest, i8 zeroext %ptr_write_val) {
entry:
  %call = call spir_func zeroext i8 @from_buf(i8 zeroext %ptr_write_val)
  %i1trunc = trunc i8 %call to i1
  %0 = load i8 addrspace(1)*, i8 addrspace(1)* addrspace(1)* @p_var, align 8
  %frombool = select i1 %i1trunc, i8 1, i8 0
  store volatile i8 %frombool, i8 addrspace(1)* %0, align 1
  %1 = load i8, i8 addrspace(1)* @var, align 1
  %2 = and i8 %1, 1
  %tobool = icmp ne i8 %2, 0
  %i1promo = zext i1 %tobool to i8
  %call1 = call spir_func zeroext i8 @to_buf(i8 zeroext %i1promo)
  %arrayidx = getelementptr inbounds i8, i8 addrspace(1)* %dest, i64 0
  store i8 %call1, i8 addrspace(1)* %arrayidx, align 1
  %3 = load i8, i8 addrspace(1)* @g_var, align 1
  %4 = and i8 %3, 1
  %tobool2 = icmp ne i8 %4, 0
  %i1promo1 = zext i1 %tobool2 to i8
  %call3 = call spir_func zeroext i8 @to_buf(i8 zeroext %i1promo1)
  %arrayidx4 = getelementptr inbounds i8, i8 addrspace(1)* %dest, i64 1
  store i8 %call3, i8 addrspace(1)* %arrayidx4, align 1
  %5 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(1)* @a_var, i64 0, i64 0
  %6 = load i8, i8 addrspace(1)* %5, align 1
  %7 = and i8 %6, 1
  %tobool5 = icmp ne i8 %7, 0
  %i1promo2 = zext i1 %tobool5 to i8
  %call6 = call spir_func zeroext i8 @to_buf(i8 zeroext %i1promo2)
  %arrayidx7 = getelementptr inbounds i8, i8 addrspace(1)* %dest, i64 2
  store i8 %call6, i8 addrspace(1)* %arrayidx7, align 1
  %8 = getelementptr inbounds [2 x i8], [2 x i8] addrspace(1)* @a_var, i64 0, i64 1
  %9 = load i8, i8 addrspace(1)* %8, align 1
  %10 = and i8 %9, 1
  %tobool8 = icmp ne i8 %10, 0
  %i1promo3 = zext i1 %tobool8 to i8
  %call9 = call spir_func zeroext i8 @to_buf(i8 zeroext %i1promo3)
  %arrayidx10 = getelementptr inbounds i8, i8 addrspace(1)* %dest, i64 3
  store i8 %call9, i8 addrspace(1)* %arrayidx10, align 1
  ret void
}
