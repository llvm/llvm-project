; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV-DAG: OpDecorate %[[#SC3:]] SpecId 3
; CHECK-SPIRV-DAG: OpDecorate %[[#SC4:]] SpecId 4
; CHECK-SPIRV-DAG: OpDecorate %[[#SC6:]] SpecId 6
; CHECK-SPIRV-DAG: OpDecorate %[[#SC7:]] SpecId 7
; CHECK-SPIRV-DAG: OpDecorate %[[#SC10:]] SpecId 10
; CHECK-SPIRV-DAG: OpDecorate %[[#SC11:]] SpecId 11

; CHECK-SPIRV-DAG: %[[#Int:]] = OpTypeInt 32
; CHECK-SPIRV-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-SPIRV-DAG: %[[#StructA:]] = OpTypeStruct %[[#Int]] %[[#Float]]
; CHECK-SPIRV-DAG: %[[#Array:]] = OpTypeArray %[[#StructA]] %[[#]]
; CHECK-SPIRV-DAG: %[[#Vector:]] = OpTypeVector %[[#Int]] 2
; CHECK-SPIRV-DAG: %[[#Struct:]] = OpTypeStruct %[[#Vector]]
; CHECK-SPIRV-DAG: %[[#POD_TYPE:]] = OpTypeStruct %[[#Array]] %[[#Struct]]

%struct._ZTS3POD.POD = type { [2 x %struct._ZTS1A.A], %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" }
%struct._ZTS1A.A = type { i32, float }
%"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" = type { <2 x i32> }
%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$_ZTS4Test = comdat any

define weak_odr dso_local spir_kernel void @_ZTS4Test(%struct._ZTS3POD.POD addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr comdat {
entry:
  %ref.tmp.i = alloca %struct._ZTS3POD.POD, align 8
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = load i64, i64* %0, align 8
  %add.ptr.i = getelementptr inbounds %struct._ZTS3POD.POD, %struct._ZTS3POD.POD addrspace(1)* %_arg_, i64 %1
  %2 = bitcast %struct._ZTS3POD.POD* %ref.tmp.i to i8*
  call void @llvm.lifetime.start.p0i8(i64 24, i8* nonnull %2)
  %3 = addrspacecast %struct._ZTS3POD.POD* %ref.tmp.i to %struct._ZTS3POD.POD addrspace(4)*

  %4 = call i32 @_Z20__spirv_SpecConstantii(i32 3, i32 1)
; CHECK-SPIRV-DAG: %[[#SC3]] = OpSpecConstant %[[#Int]] 1

  %5 = call float @_Z20__spirv_SpecConstantif(i32 4, float 0.000000e+00)
; CHECK-SPIRV-DAG: %[[#SC4]] = OpSpecConstant %[[#Float]] 0

  %6 = call %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32 %4, float %5)
; CHECK-SPIRV-DAG: %[[#SC_StructA0:]] = OpSpecConstantComposite %[[#StructA]] %[[#SC3]] %[[#SC4]]

  %7 = call i32 @_Z20__spirv_SpecConstantii(i32 6, i32 35)
; CHECK-SPIRV-DAG: %[[#SC6]] = OpSpecConstant %[[#Int]] 35

  %8 = call float @_Z20__spirv_SpecConstantif(i32 7, float 0.000000e+00)
; CHECK-SPIRV-DAG: %[[#SC7]] = OpSpecConstant %[[#Float]] 0

  %9 = call %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32 %7, float %8)
; CHECK-SPIRV-DAG: %[[#SC_StructA1:]] = OpSpecConstantComposite %[[#StructA]] %[[#SC6]] %[[#SC7]]

  %10 = call [2 x %struct._ZTS1A.A] @_Z29__spirv_SpecConstantCompositestruct._ZTS1A.Astruct._ZTS1A.A(%struct._ZTS1A.A %6, %struct._ZTS1A.A %9)
; CHECK-SPIRV-DAG: %[[#SC_Array:]] = OpSpecConstantComposite %[[#Array]] %[[#SC_StructA0]] %[[#SC_StructA1]]

  %11 = call i32 @_Z20__spirv_SpecConstantii(i32 10, i32 45)
; CHECK-SPIRV-DAG: %[[#SC10]] = OpSpecConstant %[[#Int]] 45

  %12 = call i32 @_Z20__spirv_SpecConstantii(i32 11, i32 55)
; CHECK-SPIRV-DAG: %[[#SC11]] = OpSpecConstant %[[#Int]] 55

  %13 = call <2 x i32> @_Z29__spirv_SpecConstantCompositeii(i32 %11, i32 %12)
; CHECK-SPIRV-DAG: %[[#SC_Vector:]] = OpSpecConstantComposite %[[#Vector]] %[[#SC10]] %[[#SC11]]

  %14 = call %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" @_Z29__spirv_SpecConstantCompositeDv2_i(<2 x i32> %13)
; CHECK-SPIRV-DAG: %[[#SC_Struct:]] = OpSpecConstantComposite %[[#Struct]] %[[#SC_Vector]]

  %15 = call %struct._ZTS3POD.POD @"_Z29__spirv_SpecConstantCompositeAstruct._ZTS1A.Aclass._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec"([2 x %struct._ZTS1A.A] %10, %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" %14)
; CHECK-SPIRV-DAG: %[[#SC_POD:]] = OpSpecConstantComposite %[[#POD_TYPE]] %[[#SC_Array]] %[[#SC_Struct]]

  store %struct._ZTS3POD.POD %15, %struct._ZTS3POD.POD addrspace(4)* %3, align 8
; CHECK-SPIRV-DAG: OpStore %[[#]] %[[#SC_POD]]

  %16 = bitcast %struct._ZTS3POD.POD addrspace(1)* %add.ptr.i to i8 addrspace(1)*
  %17 = addrspacecast i8 addrspace(1)* %16 to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* align 8 dereferenceable(24) %17, i8* nonnull align 8 dereferenceable(24) %2, i64 24, i1 false)
  call void @llvm.lifetime.end.p0i8(i64 24, i8* nonnull %2)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

declare void @llvm.memcpy.p4i8.p0i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)

declare i32 @_Z20__spirv_SpecConstantii(i32, i32)

declare float @_Z20__spirv_SpecConstantif(i32, float)

declare %struct._ZTS1A.A @_Z29__spirv_SpecConstantCompositeif(i32, float)

declare [2 x %struct._ZTS1A.A] @_Z29__spirv_SpecConstantCompositestruct._ZTS1A.Astruct._ZTS1A.A(%struct._ZTS1A.A, %struct._ZTS1A.A)

declare <2 x i32> @_Z29__spirv_SpecConstantCompositeii(i32, i32)

declare %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec" @_Z29__spirv_SpecConstantCompositeDv2_i(<2 x i32>)

declare %struct._ZTS3POD.POD @"_Z29__spirv_SpecConstantCompositeAstruct._ZTS1A.Aclass._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec"([2 x %struct._ZTS1A.A], %"class._ZTSN2cl4sycl3vecIiLi2EEE.cl::sycl::vec")
