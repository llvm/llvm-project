; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define i32 @test_getdimensions_no_mips() {
  ; CHECK: [[HANDLE1:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, 
  ; CHECK-NEXT: [[ANNOT_HANDLE1:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[HANDLE1]]
  %handle1 = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NEXT: [[RETVAL1:%.*]] = call %dx.types.Dimensions @dx.op.getDimensions(i32 72, %dx.types.Handle [[ANNOT_HANDLE1]], i32 undef)
  %1 = call { i32, i32, i32, i32 } @llvm.dx.resource.getdimensions.tdx.RawBuffer_i32_1_0t(target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %handle1, i32 poison)
  
  ; CHECK-NEXT: %[[DIM1:.*]] = extractvalue %dx.types.Dimensions [[RETVAL1]], 0
  %2 = extractvalue { i32, i32, i32, i32 } %1, 0
  
  ; CHECK-NEXT: ret i32 %[[DIM1]]
  ret i32 %2
}


define i32 @test_getdimensions_with_0_mips() {
  ; CHECK: [[HANDLE2:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, 
  ; CHECK-NEXT: [[ANNOT_HANDLE2:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[HANDLE2]]
  %handle1 = call target("dx.RawBuffer", float, 0, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)

  ; CHECK-NEXT: [[RETVAL2:%.*]] = call %dx.types.Dimensions @dx.op.getDimensions(i32 72, %dx.types.Handle [[ANNOT_HANDLE2]], i32 0)
  %1 = call { i32, i32, i32, i32 } @llvm.dx.resource.getdimensions.tdx.RawBuffer_i32_1_0t(target("dx.RawBuffer", float, 0, 0) %handle1, i32 0)
  
  ; CHECK-NEXT: %[[DIM2:.*]] = extractvalue %dx.types.Dimensions [[RETVAL2]], 0
  %2 = extractvalue { i32, i32, i32, i32 } %1, 0
  
  ; CHECK-NEXT: ret i32 %[[DIM2]]
  ret i32 %2
}
