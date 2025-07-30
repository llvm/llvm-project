; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; This test exercises the lowering of the intrinsic @llvm.uadd.with.overflow.i32 to the UAddc DXIL op

; CHECK-DAG: [[DX_TYPES_I32C:%dx\.types\.i32c]] = type { i32, i1 }

; NOTE: The uint2 overload of AddUint64 HLSL uses @llvm.uadd.with.overflow.i32, resulting in one UAddc op
define noundef i32 @test_UAddc(i32 noundef %a, i32 noundef %b) {
; CHECK-LABEL: define noundef i32 @test_UAddc(
; CHECK-SAME: i32 noundef [[A:%.*]], i32 noundef [[B:%.*]]) {
; CHECK-NEXT:    [[UADDC:%.*]] = call [[DX_TYPES_I32C]] @dx.op.binaryWithCarryOrBorrow.i32(i32 44, i32 [[A]], i32 [[B]]) #[[ATTR0:[0-9]+]]
; CHECK-NEXT:    [[CARRY:%.*]] = extractvalue [[DX_TYPES_I32C]] [[UADDC]], 1
; CHECK-NEXT:    [[SUM:%.*]] = extractvalue [[DX_TYPES_I32C]] [[UADDC]], 0
; CHECK-NEXT:    [[CARRY_ZEXT:%.*]] = zext i1 [[CARRY]] to i32
; CHECK-NEXT:    [[RESULT:%.*]] = add i32 [[SUM]], [[CARRY_ZEXT]]
; CHECK-NEXT:    ret i32 [[RESULT]]
;
  %uaddc = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  %carry = extractvalue { i32, i1 } %uaddc, 1
  %sum = extractvalue { i32, i1 } %uaddc, 0
  %carry_zext = zext i1 %carry to i32
  %result = add i32 %sum, %carry_zext
  ret i32 %result
}

; NOTE: The uint4 overload of AddUint64 HLSL uses @llvm.uadd.with.overflow.v2i32, resulting in two UAddc ops after scalarization
define noundef <2 x i32> @test_UAddc_vec2(<2 x i32> noundef %a, <2 x i32> noundef %b) {
; CHECK-LABEL: define noundef <2 x i32> @test_UAddc_vec2(
; CHECK-SAME: <2 x i32> noundef [[A:%.*]], <2 x i32> noundef [[B:%.*]]) {
; CHECK-NEXT:    [[A_I0:%.*]] = extractelement <2 x i32> [[A]], i64 0
; CHECK-NEXT:    [[B_I0:%.*]] = extractelement <2 x i32> [[B]], i64 0
; CHECK-NEXT:    [[UADDC_I0:%.*]] = call [[DX_TYPES_I32C]] @dx.op.binaryWithCarryOrBorrow.i32(i32 44, i32 [[A_I0]], i32 [[B_I0]]) #[[ATTR0]]
; CHECK-NEXT:    [[A_I1:%.*]] = extractelement <2 x i32> [[A]], i64 1
; CHECK-NEXT:    [[B_I1:%.*]] = extractelement <2 x i32> [[B]], i64 1
; CHECK-NEXT:    [[UADDC_I1:%.*]] = call [[DX_TYPES_I32C]] @dx.op.binaryWithCarryOrBorrow.i32(i32 44, i32 [[A_I1]], i32 [[B_I1]]) #[[ATTR0]]
; CHECK-NEXT:    [[CARRY_ELEM0:%.*]] = extractvalue [[DX_TYPES_I32C]] [[UADDC_I0]], 1
; CHECK-NEXT:    [[CARRY_ELEM1:%.*]] = extractvalue [[DX_TYPES_I32C]] [[UADDC_I1]], 1
; CHECK-NEXT:    [[SUM_ELEM0:%.*]] = extractvalue [[DX_TYPES_I32C]] [[UADDC_I0]], 0
; CHECK-NEXT:    [[SUM_ELEM1:%.*]] = extractvalue [[DX_TYPES_I32C]] [[UADDC_I1]], 0
; CHECK-NEXT:    [[CARRY_ZEXT_I0:%.*]] = zext i1 [[CARRY_ELEM0]] to i32
; CHECK-NEXT:    [[CARRY_ZEXT_I1:%.*]] = zext i1 [[CARRY_ELEM1]] to i32
; CHECK-NEXT:    [[RESULT_I0:%.*]] = add i32 [[SUM_ELEM0]], [[CARRY_ZEXT_I0]]
; CHECK-NEXT:    [[RESULT_I1:%.*]] = add i32 [[SUM_ELEM1]], [[CARRY_ZEXT_I1]]
; CHECK-NEXT:    [[RESULT_UPTO0:%.*]] = insertelement <2 x i32> poison, i32 [[RESULT_I0]], i64 0
; CHECK-NEXT:    [[RESULT:%.*]] = insertelement <2 x i32> [[RESULT_UPTO0]], i32 [[RESULT_I1]], i64 1
; CHECK-NEXT:    ret <2 x i32> [[RESULT]]
;
  %uaddc = call { <2 x i32>, <2 x i1> } @llvm.uadd.with.overflow.v2i32(<2 x i32> %a, <2 x i32> %b)
  %carry = extractvalue { <2 x i32>, <2 x i1> } %uaddc, 1
  %sum = extractvalue { <2 x i32>, <2 x i1> } %uaddc, 0
  %carry_zext = zext <2 x i1> %carry to <2 x i32>
  %result = add <2 x i32> %sum, %carry_zext
  ret <2 x i32> %result
}

define noundef i32 @test_UAddc_insert(i32 noundef %a, i32 noundef %b) {
; CHECK-LABEL: define noundef i32 @test_UAddc_insert(
; CHECK-SAME: i32 noundef [[A:%.*]], i32 noundef [[B:%.*]]) {
; CHECK-NEXT:    [[UADDC:%.*]] = call [[DX_TYPES_I32C]] @dx.op.binaryWithCarryOrBorrow.i32(i32 44, i32 [[A]], i32 [[B]]) #[[ATTR0]]
; CHECK-NEXT:    [[UNUSED:%.*]] = insertvalue [[DX_TYPES_I32C]] [[UADDC]], i32 [[A]], 0
; CHECK-NEXT:    [[RESULT:%.*]] = extractvalue [[DX_TYPES_I32C]] [[UADDC]], 0
; CHECK-NEXT:    ret i32 [[RESULT]]
;
  %uaddc = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %a, i32 %b)
  insertvalue { i32, i1 } %uaddc, i32 %a, 0
  %result = extractvalue { i32, i1 } %uaddc, 0
  ret i32 %result
}

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32)

