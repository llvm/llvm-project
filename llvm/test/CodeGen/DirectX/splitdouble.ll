; RUN: opt -passes='function(scalarizer)' -S -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,NOLOWER
; RUN: opt -passes='function(scalarizer),module(dxil-op-lower)' -S -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s --check-prefixes=CHECK,WITHLOWER

define i32 @test_scalar(double noundef %D) {
; CHECK-LABEL: define i32 @test_scalar(
; CHECK-SAME: double noundef [[D:%.*]]) {
; NOLOWER-NEXT:    [[HLSL_ASUINT_I0:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[D]])
; WITHLOWER-NEXT:  [[HLSL_ASUINT_I0:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double [[D]]) #[[#ATTR:]]
; NOLOWER-NEXT:    [[EV1:%.*]] = extractvalue { i32, i32 } [[HLSL_ASUINT_I0]], 0
; NOLOWER-NEXT:    [[EV2:%.*]] = extractvalue { i32, i32 } [[HLSL_ASUINT_I0]], 1
; WITHLOWER-NEXT:  [[EV1:%.*]] = extractvalue %dx.types.splitdouble [[HLSL_ASUINT_I0]], 0
; WITHLOWER-NEXT:  [[EV2:%.*]] = extractvalue %dx.types.splitdouble [[HLSL_ASUINT_I0]], 1
; CHECK-NEXT:      [[ADD:%.*]] = add i32 [[EV1]], [[EV2]]
; CHECK-NEXT:      ret i32 [[ADD]]
;
  %hlsl.splitdouble = call { i32, i32 } @llvm.dx.splitdouble.i32(double %D)
  %1 = extractvalue { i32, i32 } %hlsl.splitdouble, 0
  %2 = extractvalue { i32, i32 } %hlsl.splitdouble, 1
  %add = add i32 %1, %2
  ret i32 %add
}


define void @test_vector_double_split_void(<2 x double> noundef %d) {
; CHECK-LABEL: define void @test_vector_double_split_void(
; CHECK-SAME: <2 x double> noundef [[D:%.*]]) {
; CHECK-NEXT:      [[D_I0:%.*]] = extractelement <2 x double> [[D]], i64 0
; NOLOWER-NEXT:    [[HLSL_ASUINT_I0:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[D_I0]])
; WITHLOWER-NEXT:  [[HLSL_ASUINT_I0:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double [[D_I0]]) #[[#ATTR]]
; CHECK-NEXT:      [[D_I1:%.*]] = extractelement <2 x double> [[D]], i64 1
; NOLOWER-NEXT:    [[HLSL_ASUINT_I1:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[D_I1]])
; WITHLOWER-NEXT:  [[HLSL_ASUINT_I1:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double [[D_I1]]) #[[#ATTR]]
; CHECK-NEXT:      ret void
;
  %hlsl.asuint = call { <2 x i32>, <2 x i32> }  @llvm.dx.splitdouble.v2i32(<2 x double> %d)
  ret void
}

define noundef <3 x i32> @test_vector_double_split(<3 x double> noundef %d) {
; CHECK-LABEL: define noundef <3 x i32> @test_vector_double_split(
; CHECK-SAME: <3 x double> noundef [[D:%.*]]) {
; CHECK-NEXT:      [[D_I0:%.*]] = extractelement <3 x double> [[D]], i64 0
; NOLOWER-NEXT:    [[HLSL_ASUINT_I0:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[D_I0]])
; WITHLOWER-NEXT:  [[HLSL_ASUINT_I0:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double [[D_I0]]) #[[#ATTR]]
; CHECK-NEXT:      [[D_I1:%.*]] = extractelement <3 x double> [[D]], i64 1
; NOLOWER-NEXT:    [[HLSL_ASUINT_I1:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[D_I1]])
; WITHLOWER-NEXT:  [[HLSL_ASUINT_I1:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double [[D_I1]]) #[[#ATTR]]
; CHECK-NEXT:      [[D_I2:%.*]] = extractelement <3 x double> [[D]], i64 2
; NOLOWER-NEXT:    [[HLSL_ASUINT_I2:%.*]] = call { i32, i32 } @llvm.dx.splitdouble.i32(double [[D_I2]])
; WITHLOWER-NEXT:  [[HLSL_ASUINT_I2:%.*]] = call %dx.types.splitdouble @dx.op.splitDouble.f64(i32 102, double [[D_I2]]) #[[#ATTR]]
; NOLOWER-NEXT:    [[DOTELEM0:%.*]] = extractvalue { i32, i32 } [[HLSL_ASUINT_I0]], 0
; WITHLOWER-NEXT:  [[DOTELEM0:%.*]] = extractvalue %dx.types.splitdouble [[HLSL_ASUINT_I0]], 0
; NOLOWER-NEXT:    [[DOTELEM01:%.*]] = extractvalue { i32, i32 } [[HLSL_ASUINT_I1]], 0
; WITHLOWER-NEXT:  [[DOTELEM01:%.*]] = extractvalue %dx.types.splitdouble [[HLSL_ASUINT_I1]], 0
; NOLOWER-NEXT:    [[DOTELEM02:%.*]] = extractvalue { i32, i32 } [[HLSL_ASUINT_I2]], 0
; WITHLOWER-NEXT:  [[DOTELEM02:%.*]] = extractvalue %dx.types.splitdouble [[HLSL_ASUINT_I2]], 0
; NOLOWER-NEXT:    [[DOTELEM1:%.*]] = extractvalue { i32, i32 } [[HLSL_ASUINT_I0]], 1
; WITHLOWER-NEXT:  [[DOTELEM1:%.*]] = extractvalue %dx.types.splitdouble [[HLSL_ASUINT_I0]], 1
; NOLOWER-NEXT:    [[DOTELEM13:%.*]] = extractvalue { i32, i32 } [[HLSL_ASUINT_I1]], 1
; WITHLOWER-NEXT:  [[DOTELEM13:%.*]] = extractvalue %dx.types.splitdouble [[HLSL_ASUINT_I1]], 1
; NOLOWER-NEXT:    [[DOTELEM14:%.*]] = extractvalue { i32, i32 } [[HLSL_ASUINT_I2]], 1
; WITHLOWER-NEXT:  [[DOTELEM14:%.*]] = extractvalue %dx.types.splitdouble [[HLSL_ASUINT_I2]], 1
; CHECK-NEXT:      [[DOTI0:%.*]] = add i32 [[DOTELEM0]], [[DOTELEM1]]
; CHECK-NEXT:      [[DOTI1:%.*]] = add i32 [[DOTELEM01]], [[DOTELEM13]]
; CHECK-NEXT:      [[DOTI2:%.*]] = add i32 [[DOTELEM02]], [[DOTELEM14]]
; CHECK-NEXT:      [[DOTUPTO015:%.*]] = insertelement <3 x i32> poison, i32 [[DOTI0]], i64 0
; CHECK-NEXT:      [[DOTUPTO116:%.*]] = insertelement <3 x i32> [[DOTUPTO015]], i32 [[DOTI1]], i64 1
; CHECK-NEXT:      [[TMP1:%.*]] = insertelement <3 x i32> [[DOTUPTO116]], i32 [[DOTI2]], i64 2
; CHECK-NEXT:      ret <3 x i32> [[TMP1]]
;
  %hlsl.asuint = call { <3 x i32>, <3 x i32> }  @llvm.dx.splitdouble.v3i32(<3 x double> %d)
  %1 = extractvalue { <3 x i32>, <3 x i32> } %hlsl.asuint, 0
  %2 = extractvalue { <3 x i32>, <3 x i32> } %hlsl.asuint, 1
  %3 = add <3 x i32> %1, %2
  ret <3 x i32> %3
}

; WITHLOWER: attributes #[[#ATTR]] = {{{.*}} memory(none) {{.*}}}
