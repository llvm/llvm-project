; RUN: opt -S -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

define i32 @removal_only_test(i32 %a) {
  ; CHECK-LABEL: define i32 @removal_only_test(
  ; CHECK-SAME: i32 [[A:%.*]]) {
  ; CHECK: [[SHL:%.*]] = shl i32 [[A]], 24
  ; CHECK: [[SIGNED:%.*]] = ashr i32 [[SHL]], 24
  ; CHECK: ret i32 [[SIGNED]]
  %1 = trunc nsw i32 %a to i8
  %3 = sext i8 %1 to i32
  ret i32 %3
}

define i32 @i8trunc(float %0) #0 {
  ; CHECK-LABEL: define i32 @i8trunc(
  ; CHECK-NOT: i8
  ; CHECK-NEXT: [[CONVERT:%.*]] = fptosi float %0 to i32
  ; CHECK-NEXT: [[INITIAL_REM:%.*]] = srem i32 [[CONVERT]], 8
  ; CHECK-NEXT: [[ADD:%.*]] = add nsw i32 [[INITIAL_REM]], 1
  ; CHECK-NEXT: [[SREM:%.*]] = srem i32 [[ADD]], 8
  ; CHECK-NEXT: [[SUB:%.*]] = sub i32 [[SREM]], 1
  ; CHECK-NEXT: [[MUL:%.*]] = mul i32 [[SUB]], 1
  ; CHECK-NEXT: [[MUL_UNSIGNED:%.*]] = and i32 [[MUL]], 255
  ; CHECK-NEXT: [[UDIV:%.*]] = udiv i32 [[MUL_UNSIGNED]], 1
  ; CHECK-NEXT: [[UDIV_SHL:%.*]] = shl i32 [[UDIV]], 24
  ; CHECK-NEXT: [[UDIV_SIGNED:%.*]] = ashr i32 [[UDIV_SHL]], 24
  ; CHECK-NEXT: [[SDIV:%.*]] = sdiv i32 [[UDIV_SIGNED]], 1
  ; CHECK-NEXT: [[SDIV_UNSIGNED:%.*]] = and i32 [[SDIV]], 255
  ; CHECK-NEXT: [[UREM:%.*]] = urem i32 [[SDIV_UNSIGNED]], 1
  ; CHECK-NEXT: [[AND:%.*]] = and i32 [[UREM]], 1
  ; CHECK-NEXT: [[OR:%.*]] = or i32 [[AND]], 1
  ; CHECK-NEXT: [[XOR:%.*]] = xor i32 [[OR]], 1
  ; CHECK-NEXT: [[SHL:%.*]] = shl i32 [[XOR]], 1
  ; CHECK-NEXT: [[LSHR:%.*]] = lshr i32 [[SHL]], 1
  ; CHECK-NEXT: [[ASHR:%.*]] = ashr i32 [[LSHR]], 1
  ; CHECK-NEXT: ret i32 [[ASHR]]
  
  %2 = fptosi float %0 to i32
  %3 = srem i32 %2, 8
  %4 = trunc nsw i32 %3 to i8
  %5 = add nsw i8 %4, 1
  %6 = srem i8 %5, 8
  %7 = sub i8 %6, 1
  %8 = mul i8 %7, 1
  %9 = udiv i8 %8, 1
  %10 = sdiv i8 %9, 1
  %11 = urem i8 %10, 1
  %12 = and i8 %11, 1
  %13 = or i8 %12, 1
  %14 = xor i8 %13, 1
  %15 = shl i8 %14, 1
  %16 = lshr i8 %15, 1
  %17 = ashr i8 %16, 1
  %18 = sext i8 %17 to i32
  ret i32 %18
}

define i32 @cast_removal_test(i32 %a) {
  ; CHECK-LABEL: define i32 @cast_removal_test(
  ; CHECK-SAME: i32 [[A:%.*]]) {
  ; CHECK-NOT: trunc
  ; CHECK-NOT: zext i8
  ; CHECK-NOT: sext i8
  ; CHECK: [[MASKED:%.*]] = and i32 [[A]], 255
  ; CHECK: [[SHL:%.*]] = shl i32 [[A]], 24
  ; CHECK: [[SIGNED:%.*]] = ashr i32 [[SHL]], 24
  ; CHECK: add i32 [[MASKED]], [[SIGNED]]
  %1 = trunc nsw i32 %a to i8
  %2 = zext i8 %1 to i32
  %3 = sext i8 %1 to i32
  %4 = add i32 %2, %3
  ret i32 %4
}

define i1 @trunc_cmp_test(i32 %a, i32 %b) {
  ; CHECK-LABEL: define i1 @trunc_cmp_test(
  ; CHECK-SAME: i32 [[A:%.*]], i32 [[B:%.*]]) {
  ; CHECK: [[ASHL:%.*]] = shl i32 [[A]], 24
  ; CHECK: [[ASIGNED:%.*]] = ashr i32 [[ASHL]], 24
  ; CHECK: [[BSHL:%.*]] = shl i32 [[B]], 24
  ; CHECK: [[BSIGNED:%.*]] = ashr i32 [[BSHL]], 24
  ; CHECK: icmp slt i32 [[ASIGNED]], [[BSIGNED]]
  ; CHECK: icmp sgt i32
  %1 = trunc nsw i32 %a to i8
  %2 = trunc nsw i32 %b to i8
  %3 = icmp slt i8 %1, %2
  %4 = icmp sgt i8 %1, %2
  %5 = and i1 %3, %4
  ret i1 %5
}

define i32 @first_operand_imm_test(i32 %a) {
  ; CHECK-LABEL: define i32 @first_operand_imm_test(
  ; CHECK-SAME: i32 [[A:%.*]]) {
  ; CHECK-NOT: trunc
  ; CHECK: sub i32 0, [[A]]
  ; CHECK-NOT: sext i8
  %1 = trunc nsw i32 %a to i8
  %2 = sub i8 0, %1
  %3 = sext i8 %2 to i32
  ret i32 %3
}

define i16 @i16_test(i16 %a) {
  ; CHECK-LABEL: define i16 @i16_test(
  ; CHECK-SAME: i16 [[A:%.*]]) {
  ; CHECK: [[EXT:%.*]] = zext i16 [[A]] to i32
  ; CHECK: sub i32 0, [[EXT]]
  ; CHECK-NOT: sext i8
  %1 = trunc nsw i16 %a to i8
  %2 = sub i8 0, %1
  %3 = sext i8 %2 to i16
  ret i16 %3
}

define i32 @all_imm() {
  ; CHECK-LABEL: define i32 @all_imm(
  ; CHECK-NOT: sext i8
  ; CHECK: ret i32 -1
  %1 = sub i8 0, 1
  %2 = sext i8 %1 to i32
  ret i32 %2
}

define i32 @scalar_i8_geps() {
  ; CHECK-LABEL: define i32 @scalar_i8_geps(
  ; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca i32, align 4
  ; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [1 x i32], ptr [[ALLOCA]], i32 0, i32 0
  ; CHECK:         [[LOAD:%.*]] = load i32, ptr [[GEP]], align 4
  ; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[LOAD]], 24
  ; CHECK-NEXT:    [[SIGNED:%.*]] = ashr i32 [[SHL]], 24
  ; CHECK-NEXT:    ret i32 [[SIGNED]]
    %1 = alloca i8, align 4
    %2 = getelementptr inbounds nuw i8, ptr %1, i32 0
    %3 = load i8, ptr %2
    %4 = sext i8 %3 to i32
    ret i32 %4
}

define i32 @i8_geps_index0() {
  ; CHECK-LABEL: define i32 @i8_geps_index0(
  ; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca [2 x i32], align 8
  ; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [2 x i32], ptr [[ALLOCA]], i32 0, i32 0
  ; CHECK:         [[LOAD:%.*]] = load i32, ptr [[GEP]], align 4
  ; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[LOAD]], 24
  ; CHECK-NEXT:    [[SIGNED:%.*]] = ashr i32 [[SHL]], 24
  ; CHECK-NEXT:    ret i32 [[SIGNED]]
  %1 = alloca [2 x i32], align 8
  %2 = load i8, ptr %1
  %3 = sext i8 %2 to i32
  ret i32 %3
}

define i32 @i8_geps_index1() {
  ; CHECK-LABEL: define i32 @i8_geps_index1(
  ; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca [2 x i32], align 8
  ; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [2 x i32], ptr [[ALLOCA]], i32 0, i32 1
  ; CHECK:         [[LOAD:%.*]] = load i32, ptr [[GEP]], align 4
  ; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[LOAD]], 24
  ; CHECK-NEXT:    [[SIGNED:%.*]] = ashr i32 [[SHL]], 24
  ; CHECK-NEXT:    ret i32 [[SIGNED]]
  %1 = alloca [2 x i32], align 8
  %2 = getelementptr inbounds nuw i8, ptr %1, i32 4
  %3 = load i8, ptr %2
  %4 = sext i8 %3 to i32
  ret i32 %4
}

define i32 @i8_gep_store() {
  ; CHECK-LABEL: define i32 @i8_gep_store(
  ; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca [2 x i32], align 8
  ; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [2 x i32], ptr [[ALLOCA]], i32 0, i32 0
  ; CHECK-NEXT:    store i32 0, ptr [[GEP]], align 4
  ; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds nuw [2 x i32], ptr [[ALLOCA]], i32 0, i32 1
  ; CHECK-NEXT:    store i32 1, ptr [[GEP]], align 4
  ; CHECK:         [[LOAD:%.*]] = load i32, ptr [[GEP]], align 4
  ; CHECK-NEXT:    [[SHL:%.*]] = shl i32 [[LOAD]], 24
  ; CHECK-NEXT:    [[SIGNED:%.*]] = ashr i32 [[SHL]], 24
  ; CHECK-NEXT:    ret i32 [[SIGNED]]
  %1 = alloca [2 x i32], align 8
  store i8 0, ptr %1
  %2 = getelementptr inbounds nuw i8, ptr %1, i32 4
  store i8 1, ptr %2
  %3 = load i8, ptr %2
  %4 = sext i8 %3 to i32
  ret i32 %4
}

@g = local_unnamed_addr addrspace(3) global [2 x float] zeroinitializer, align 4
define float @i8_gep_global_index() {
  ; CHECK-LABEL: define float @i8_gep_global_index(
  ; CHECK-NEXT: [[LOAD:%.*]] = load float, ptr addrspace(3) getelementptr inbounds nuw ([2 x float], ptr addrspace(3) @g, i32 0, i32 1), align 4
  ; CHECK-NEXT:    ret float [[LOAD]]
  %1 = getelementptr inbounds nuw i8, ptr addrspace(3) @g, i32 4
  %2 = load float, ptr addrspace(3) %1, align 4
  ret float %2
}

define float @i8_gep_global_constexpr() {
  ; CHECK-LABEL: define float @i8_gep_global_constexpr(
  ; CHECK-NEXT: [[LOAD:%.*]] = load float, ptr addrspace(3) getelementptr inbounds nuw ([2 x float], ptr addrspace(3) @g, i32 0, i32 1), align 4
  ; CHECK-NEXT: ret float [[LOAD]]
  %1 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @g, i32 4), align 4
  ret float %1
}
