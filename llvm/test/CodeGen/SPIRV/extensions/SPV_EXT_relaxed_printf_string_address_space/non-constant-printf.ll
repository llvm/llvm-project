; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_EXT_relaxed_printf_string_address_space %s -o - | FileCheck %s
; RUN: not llc -O0 -mtriple=spirv32-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK: OpExtension "SPV_EXT_relaxed_printf_string_address_space"
; CHECK: %[[#ExtInstSetId:]] = OpExtInstImport "OpenCL.std"
; CHECK-DAG: %[[#TypeInt32Id:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#TypeInt8Id:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#TypeInt64Id:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#TypeArrayId:]] = OpTypeArray %[[#TypeInt8Id]] %[[#]]
; CHECK-DAG: %[[#ConstantStorClassGlobalPtrTy:]] = OpTypePointer UniformConstant %[[#TypeArrayId]]
; CHECK-DAG: %[[#WGStorClassGlobalPtrTy:]] = OpTypePointer Workgroup %[[#TypeArrayId]]
; CHECK-DAG: %[[#CrossWFStorClassGlobalPtrTy:]] = OpTypePointer CrossWorkgroup %[[#TypeArrayId]]
; CHECK-DAG: %[[#FunctionStorClassPtrTy:]] = OpTypePointer Function %[[#TypeInt8Id]]
; CHECK-DAG: %[[#WGStorClassPtrTy:]] = OpTypePointer Workgroup %[[#TypeInt8Id]]
; CHECK-DAG: %[[#CrossWFStorClassPtrTy:]] = OpTypePointer CrossWorkgroup %[[#TypeInt8Id]]
; CHECK: %[[#ConstantCompositeId:]] = OpConstantComposite %[[#TypeArrayId]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpVariable %[[#ConstantStorClassGlobalPtrTy]] UniformConstant %[[#ConstantCompositeId]]
; CHECK: %[[#]] = OpVariable %[[#CrossWFStorClassGlobalPtrTy]] CrossWorkgroup %[[#ConstantCompositeId]]
; CHECK: %[[#]] = OpVariable %[[#WGStorClassGlobalPtrTy]] Workgroup %[[#ConstantCompositeId]]
; CHECK: %[[#GEP1:]] = OpInBoundsPtrAccessChain %[[#FunctionStorClassPtrTy]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#TypeInt32Id]] %[[#ExtInstSetId:]] printf %[[#GEP1]]
; CHECK: %[[#GEP2:]] = OpInBoundsPtrAccessChain %[[#CrossWFStorClassPtrTy]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#TypeInt32Id]] %[[#ExtInstSetId:]] printf %[[#GEP2]]
; CHECK: %[[#GEP3:]] = OpInBoundsPtrAccessChain %[[#WGStorClassPtrTy]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#TypeInt32Id]] %[[#ExtInstSetId:]] printf %[[#GEP3]]

; CHECK-ERROR: LLVM ERROR: SPV_EXT_relaxed_printf_string_address_space is required because printf uses a format string not in constant address space.

@0 = internal unnamed_addr addrspace(2) constant [6 x i8] c"Test\0A\00", align 1
@1 = internal unnamed_addr addrspace(1) constant [6 x i8] c"Test\0A\00", align 1
@2 = internal unnamed_addr addrspace(3) constant [6 x i8] c"Test\0A\00", align 1

define spir_kernel void @test() {
  %tmp1 = alloca [6 x i8], align 1
  call void @llvm.memcpy.p0.p2.i64(ptr align 1 %tmp1, ptr addrspace(2) align 1 @0, i64 6, i1 false)
  %1 = getelementptr inbounds [6 x i8], ptr %tmp1, i32 0, i32 0
  %2 = call spir_func i32 @_Z18__spirv_ocl_printfPc(ptr %1)
  %3 = getelementptr inbounds [6 x i8], ptr addrspace(1) @1, i32 0, i32 0
  %4 = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS1c(ptr addrspace(1) %3)
  %5 = getelementptr inbounds [6 x i8], ptr addrspace(3) @2, i32 0, i32 0
  %6 = call spir_func i32 @_Z18__spirv_ocl_printfPU3AS3c(ptr addrspace(3) %5)
  ret void
}

declare spir_func i32 @_Z18__spirv_ocl_printfPc(ptr)
declare spir_func i32 @_Z18__spirv_ocl_printfPU3AS1c(ptr addrspace(1))
declare spir_func i32 @_Z18__spirv_ocl_printfPU3AS3c(ptr addrspace(3))
declare void @llvm.memcpy.p0.p2.i64(ptr captures(none), ptr addrspace(2) captures(none) readonly, i64, i1)
