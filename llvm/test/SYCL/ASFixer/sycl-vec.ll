; RUN: opt -asfix %s -S -o %t.out
; FileCheck %s --input-file %t.out
; FileCheck %s --input-file %t.out -check-prefix=CHECK-FOO
; FileCheck %s --input-file %t.out -check-prefix=CHECK-FOO1
; FileCheck %s --input-file %t.out -check-prefix=CHECK-FOO2
; FileCheck %s --input-file %t.out -check-prefix=CHECK-FOO3
source_filename = "simple-example-of-vectors.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

%"struct.std::array" = type { [1 x i32] }
%"struct.std::array.31" = type { [3 x i32] }
%"class.cl::sycl::vec" = type { <4 x float> }
%"class.cl::sycl::vec.0" = type { <3 x float> }
%"class.cl::sycl::vecdetail::SwizzleOp" = type <{ %"class.cl::sycl::vec"*, %"class.cl::sycl::vecdetail::GetOp", %"class.cl::sycl::vecdetail::GetOp", [6 x i8] }>
%"class.cl::sycl::vecdetail::GetOp" = type { i8 }

$_ZN2cl4sycl3vecIfLi4EEC2ERKS2_ = comdat any

$_ZN2cl4sycl3vecIfLi4EE1wEv = comdat any

$_ZN2cl4sycl9vecdetail9SwizzleOpINS0_3vecIfLi4EEENS1_5GetOpIfEES6_S5_JLi3EEEC2EPS4_ = comdat any

$_ZN2cl4sycl3vecIfLi3EEaSERKS2_ = comdat any

@_ZZNK2cl4sycl9vecdetail9SwizzleOpINS0_3vecIfLi4EEENS1_5GetOpIfEES6_S5_JLi3EEEcvT_IfLi1EvvEEvE4Idxs = external dso_local unnamed_addr constant %"struct.std::array", align 4
@_ZZNK2cl4sycl9vecdetail9SwizzleOpINS0_3vecIfLi4EEENS1_5GetOpIfEES6_S5_JLi3EEE8getValueEmE4Idxs = external dso_local unnamed_addr constant %"struct.std::array", align 4
@_ZZNK2cl4sycl9vecdetail9SwizzleOpINS0_3vecIfLi4EEENS1_5GetOpIfEES6_S5_JLi0ELi1ELi2EEE8getValueEmE4Idxs = external dso_local unnamed_addr constant %"struct.std::array.31", align 4

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1) #0

; Function Attrs: noinline
define dso_local spir_func void @_Z5usagePU3AS1N2cl4sycl3vecIfLi4EEEPU3AS1NS1_IfLi3EEEPU3AS3S5_(%"class.cl::sycl::vec" addrspace(1)* %ptrA, %"class.cl::sycl::vec.0" addrspace(1)* %ptrB, %"class.cl::sycl::vec.0" addrspace(3)* %atile) #1 {
entry:
  %ptrA.addr = alloca %"class.cl::sycl::vec" addrspace(1)*, align 8
  %ptrB.addr = alloca %"class.cl::sycl::vec.0" addrspace(1)*, align 8
  %atile.addr = alloca %"class.cl::sycl::vec.0" addrspace(3)*, align 8
  %in = alloca %"class.cl::sycl::vec", align 16
; CHECK:  %[[SWIZZLE_ALLOC:.*]] = alloca %"new.class.cl::sycl::vecdetail::SwizzleOp"
  %ref.tmp = alloca %"class.cl::sycl::vecdetail::SwizzleOp", align 8
  %scaled = alloca %"class.cl::sycl::vec.0", align 16
  store %"class.cl::sycl::vec" addrspace(1)* %ptrA, %"class.cl::sycl::vec" addrspace(1)** %ptrA.addr, align 8
  store %"class.cl::sycl::vec.0" addrspace(1)* %ptrB, %"class.cl::sycl::vec.0" addrspace(1)** %ptrB.addr, align 8
  store %"class.cl::sycl::vec.0" addrspace(3)* %atile, %"class.cl::sycl::vec.0" addrspace(3)** %atile.addr, align 8
  %0 = load %"class.cl::sycl::vec" addrspace(1)*, %"class.cl::sycl::vec" addrspace(1)** %ptrA.addr, align 8
  %arrayidx = getelementptr inbounds %"class.cl::sycl::vec", %"class.cl::sycl::vec" addrspace(1)* %0, i64 0
; CHECK: %[[NEW_CAST:.*]] = addrspacecast %"class.cl::sycl::vec" addrspace(1)* %{{.*}} to  %"class.cl::sycl::vec" addrspace(4)*
; CHECK: %[[NEW_CAST1:.*]] = addrspacecast %"class.cl::sycl::vec"* %{{.*}} to  %"class.cl::sycl::vec" addrspace(4)*
; CHECK: call spir_func void @new.[[FOO:.*]](%"class.cl::sycl::vec" addrspace(4)* %[[NEW_CAST1]], %"class.cl::sycl::vec" addrspace(4)* %[[NEW_CAST]])
  %1 = addrspacecast %"class.cl::sycl::vec" addrspace(1)* %arrayidx to %"class.cl::sycl::vec"*
  call spir_func void @_ZN2cl4sycl3vecIfLi4EEC2ERKS2_(%"class.cl::sycl::vec"* %in, %"class.cl::sycl::vec"* dereferenceable(16) %1)
  %2 = load %"class.cl::sycl::vec" addrspace(1)*, %"class.cl::sycl::vec" addrspace(1)** %ptrA.addr, align 8
  %arrayidx1 = getelementptr inbounds %"class.cl::sycl::vec", %"class.cl::sycl::vec" addrspace(1)* %2, i64 0
; CHECK: %[[NEW_CAST6:.*]] = addrspacecast %"class.cl::sycl::vec" addrspace(1)* %{{.*}} to  %"class.cl::sycl::vec" addrspace(4)*
; CHECK: call spir_func void @new.[[FOO2:.*]](%"new.class.cl::sycl::vecdetail::SwizzleOp"* %[[SWIZZLE_ALLOC]], %"class.cl::sycl::vec" addrspace(4)* %[[NEW_CAST6]])
  %3 = addrspacecast %"class.cl::sycl::vec" addrspace(1)* %arrayidx1 to %"class.cl::sycl::vec"*
  call spir_func void @_ZN2cl4sycl3vecIfLi4EE1wEv(%"class.cl::sycl::vecdetail::SwizzleOp"* sret %ref.tmp, %"class.cl::sycl::vec"* %3)
  %4 = load %"class.cl::sycl::vec.0" addrspace(3)*, %"class.cl::sycl::vec.0" addrspace(3)** %atile.addr, align 8
  %arrayidx3 = getelementptr inbounds %"class.cl::sycl::vec.0", %"class.cl::sycl::vec.0" addrspace(3)* %4, i64 0
; CHECK: %[[NEW_CAST2:.*]] = addrspacecast %"class.cl::sycl::vec.0" addrspace(3)* %{{.*}} to  %"class.cl::sycl::vec.0" addrspace(4)*
; CHECK: %[[NEW_CAST3:.*]] = addrspacecast %"class.cl::sycl::vec.0"* %{{.*}} to  %"class.cl::sycl::vec.0" addrspace(4)*
; CHECK:  %{{.*}} = call spir_func %"class.cl::sycl::vec.0" addrspace(4)* @new.[[FOO1:.*]](%"class.cl::sycl::vec.0" addrspace(4)* %[[NEW_CAST2]], %"class.cl::sycl::vec.0" addrspace(4)* %[[NEW_CAST3]])
  %5 = addrspacecast %"class.cl::sycl::vec.0" addrspace(3)* %arrayidx3 to %"class.cl::sycl::vec.0"*
  %call4 = call spir_func dereferenceable(16) %"class.cl::sycl::vec.0"* @_ZN2cl4sycl3vecIfLi3EEaSERKS2_(%"class.cl::sycl::vec.0"* %5, %"class.cl::sycl::vec.0"* dereferenceable(16) %scaled)
  %6 = load %"class.cl::sycl::vec.0" addrspace(3)*, %"class.cl::sycl::vec.0" addrspace(3)** %atile.addr, align 8
  %arrayidx5 = getelementptr inbounds %"class.cl::sycl::vec.0", %"class.cl::sycl::vec.0" addrspace(3)* %6, i64 0
  %7 = load %"class.cl::sycl::vec.0" addrspace(1)*, %"class.cl::sycl::vec.0" addrspace(1)** %ptrB.addr, align 8
  %arrayidx6 = getelementptr inbounds %"class.cl::sycl::vec.0", %"class.cl::sycl::vec.0" addrspace(1)* %7, i64 0
; CHECK: %[[NEW_CAST4:.*]] = addrspacecast %"class.cl::sycl::vec.0" addrspace(1)* %{{.*}} to  %"class.cl::sycl::vec.0" addrspace(4)*
; CHECK: %[[NEW_CAST5:.*]] = addrspacecast %"class.cl::sycl::vec.0" addrspace(3)* %{{.*}} to  %"class.cl::sycl::vec.0" addrspace(4)*
; CHECK:  %{{.*}} = call spir_func %"class.cl::sycl::vec.0" addrspace(4)* @new.[[FOO1]](%"class.cl::sycl::vec.0" addrspace(4)* %[[NEW_CAST4]], %"class.cl::sycl::vec.0" addrspace(4)* %[[NEW_CAST5]])
  %8 = addrspacecast %"class.cl::sycl::vec.0" addrspace(1)* %arrayidx6 to %"class.cl::sycl::vec.0"*
  %9 = addrspacecast %"class.cl::sycl::vec.0" addrspace(3)* %arrayidx5 to %"class.cl::sycl::vec.0"*
  %call7 = call spir_func dereferenceable(16) %"class.cl::sycl::vec.0"* @_ZN2cl4sycl3vecIfLi3EEaSERKS2_(%"class.cl::sycl::vec.0"* %8, %"class.cl::sycl::vec.0"* dereferenceable(16) %9)
  ret void
}

; CHECK-FOO: define linkonce_odr dso_local spir_func void @new.[[FOO]](%"class.cl::sycl::vec" addrspace(4)*, %"class.cl::sycl::vec" addrspace(4)* dereferenceable(16)) unnamed_addr #2 align 2 {
; Function Attrs: noinline nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl3vecIfLi4EEC2ERKS2_(%"class.cl::sycl::vec"* %this, %"class.cl::sycl::vec"* dereferenceable(16) %Rhs) unnamed_addr #2 comdat align 2 {
entry:
; CHECK-FOO:  %new.[[THIS_ALLOC:.*]] = alloca %"class.cl::sycl::vec" addrspace(4)*
  %this.addr = alloca %"class.cl::sycl::vec"*, align 8
; CHECK-FOO:  %new.[[RHS_ALLOC:.*]] = alloca %"class.cl::sycl::vec" addrspace(4)*
  %Rhs.addr = alloca %"class.cl::sycl::vec"*, align 8
; CHECK-FOO:  store %"class.cl::sycl::vec" addrspace(4)* %{{.*}}, %"class.cl::sycl::vec" addrspace(4)** %new.[[THIS_ALLOC]], align 8
  store %"class.cl::sycl::vec"* %this, %"class.cl::sycl::vec"** %this.addr, align 8
; CHECK-FOO:  store %"class.cl::sycl::vec" addrspace(4)* %{{.*}}, %"class.cl::sycl::vec" addrspace(4)** %new.[[RHS_ALLOC]], align 8
  store %"class.cl::sycl::vec"* %Rhs, %"class.cl::sycl::vec"** %Rhs.addr, align 8
; CHECK-FOO:  %[[THIS_LOAD:.*]] = load %"class.cl::sycl::vec" addrspace(4)*, %"class.cl::sycl::vec" addrspace(4)** %new.[[THIS_ALLOC]]
  %this1 = load %"class.cl::sycl::vec"*, %"class.cl::sycl::vec"** %this.addr, align 8
; CHECK-FOO:  %[[THIS_GEP:.*]] = getelementptr %"class.cl::sycl::vec", %"class.cl::sycl::vec" addrspace(4)* %[[THIS_LOAD]], i32 0, i32 0
  %m_Data = getelementptr inbounds %"class.cl::sycl::vec", %"class.cl::sycl::vec"* %this1, i32 0, i32 0
; CHECK-FOO:  %[[RHS_LOAD:.*]] = load %"class.cl::sycl::vec" addrspace(4)*, %"class.cl::sycl::vec" addrspace(4)** %new.[[RHS_ALLOC]]
  %0 = load %"class.cl::sycl::vec"*, %"class.cl::sycl::vec"** %Rhs.addr, align 8
; CHECK-FOO:  %[[RHS_GEP:.*]] = getelementptr %"class.cl::sycl::vec", %"class.cl::sycl::vec" addrspace(4)* %[[RHS_LOAD]], i32 0, i32 0
  %m_Data2 = getelementptr inbounds %"class.cl::sycl::vec", %"class.cl::sycl::vec"* %0, i32 0, i32 0
; CHECK-FOO:  %[[RHS_GEP_LOAD:.*]] = load <4 x float>, <4 x float> addrspace(4)* %[[RHS_GEP]]
  %1 = load <4 x float>, <4 x float>* %m_Data2, align 16
; CHECK-FOO:  store <4 x float> %[[RHS_GEP_LOAD:.*]], <4 x float> addrspace(4)* %[[THIS_GEP]], align 16
  store <4 x float> %1, <4 x float>* %m_Data, align 16
  ret void
}

; CHECK-FOO2: define linkonce_odr dso_local spir_func void @new.[[FOO2]](%"new.class.cl::sycl::vecdetail::SwizzleOp"* noalias sret, %"class.cl::sycl::vec" addrspace(4)*) #1 align 2 {
; Function Attrs: noinline optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl3vecIfLi4EE1wEv(%"class.cl::sycl::vecdetail::SwizzleOp"* noalias sret %agg.result, %"class.cl::sycl::vec"* %this) #1 comdat align 2 {
entry:
; CHECK-FOO2:  %new.[[THIS_ALLOC:.*]] = alloca %"class.cl::sycl::vec" addrspace(4)*
  %this.addr = alloca %"class.cl::sycl::vec"*, align 8
; CHECK-FOO2:  store %"class.cl::sycl::vec" addrspace(4)* %{{.*}}, %"class.cl::sycl::vec" addrspace(4)** %new.[[THIS_ALLOC]], align 8
  store %"class.cl::sycl::vec"* %this, %"class.cl::sycl::vec"** %this.addr, align 8
; CHECK-FOO2:  %[[THIS_LOAD:.*]] = load %"class.cl::sycl::vec" addrspace(4)*, %"class.cl::sycl::vec" addrspace(4)** %new.[[THIS_ALLOC]]
  %this1 = load %"class.cl::sycl::vec"*, %"class.cl::sycl::vec"** %this.addr, align 8
; CHECK-FOO2:  call spir_func void @new.[[FOO3:.*]](%"new.class.cl::sycl::vecdetail::SwizzleOp"* %{{.*}}, %"class.cl::sycl::vec" addrspace(4)* %[[THIS_LOAD]])
  call spir_func void @_ZN2cl4sycl9vecdetail9SwizzleOpINS0_3vecIfLi4EEENS1_5GetOpIfEES6_S5_JLi3EEEC2EPS4_(%"class.cl::sycl::vecdetail::SwizzleOp"* %agg.result, %"class.cl::sycl::vec"* %this1)
  ret void
}

; CHECK-FOO3: define linkonce_odr dso_local spir_func void @new.[[FOO3]](%"new.class.cl::sycl::vecdetail::SwizzleOp"*, %"class.cl::sycl::vec" addrspace(4)*) unnamed_addr #2 align 2 {
; Function Attrs: noinline nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl9vecdetail9SwizzleOpINS0_3vecIfLi4EEENS1_5GetOpIfEES6_S5_JLi3EEEC2EPS4_(%"class.cl::sycl::vecdetail::SwizzleOp"* %this, %"class.cl::sycl::vec"* %Vector) unnamed_addr #2 comdat align 2 {
entry:
; CHECK-FOO3:  %[[THIS_ALLOC:.*]] = alloca %"new.class.cl::sycl::vecdetail::SwizzleOp"*
  %this.addr = alloca %"class.cl::sycl::vecdetail::SwizzleOp"*, align 8
; CHECK-FOO3:  %[[VEC_ALLOC:.*]] = alloca %"class.cl::sycl::vec" addrspace(4)*
  %Vector.addr = alloca %"class.cl::sycl::vec"*, align 8
; CHECK-FOO3:  store %"new.class.cl::sycl::vecdetail::SwizzleOp"* %{{.*}}, %"class.cl::sycl::vecdetail::SwizzleOp"** %[[THIS_ALLOC]], align 8
  store %"class.cl::sycl::vecdetail::SwizzleOp"* %this, %"class.cl::sycl::vecdetail::SwizzleOp"** %this.addr, align 8
; CHECK-FOO3:  store %"class.cl::sycl::vec" addrspace(4)* %{{.*}}, %"class.cl::sycl::vec" addrspace(4)** %[[VEC_ALLOC]], align 8
  store %"class.cl::sycl::vec"* %Vector, %"class.cl::sycl::vec"** %Vector.addr, align 8
; CHECK-FOO3:  %[[THIS_LOAD:.*]] = load %"new.class.cl::sycl::vecdetail::SwizzleOp"*, %"new.class.cl::sycl::vecdetail::SwizzleOp"** %[[THIS_ALLOC]], align 8
  %this1 = load %"class.cl::sycl::vecdetail::SwizzleOp"*, %"class.cl::sycl::vecdetail::SwizzleOp"** %this.addr, align 8
; CHECK-FOO3:  %[[THIS_GEP:.*]] = getelementptr inbounds %"new.class.cl::sycl::vecdetail::SwizzleOp", %"new.class.cl::sycl::vecdetail::SwizzleOp"* %[[THIS_LOAD]], i32 0, i32 0
  %m_Vector = getelementptr inbounds %"class.cl::sycl::vecdetail::SwizzleOp", %"class.cl::sycl::vecdetail::SwizzleOp"* %this1, i32 0, i32 0
; CHECK-FOO3:  %[[VEC_LOAD:.*]] = load %"class.cl::sycl::vec" addrspace(4)*, %"class.cl::sycl::vec" addrspace(4)** %[[VEC_ALLOC]]
  %0 = load %"class.cl::sycl::vec"*, %"class.cl::sycl::vec"** %Vector.addr, align 8
; CHECK-FOO3:  store %"class.cl::sycl::vec" addrspace(4)* %[[VEC_LOAD]], %"class.cl::sycl::vec" addrspace(4)** %[[THIS_GEP]], align 8
  store %"class.cl::sycl::vec"* %0, %"class.cl::sycl::vec"** %m_Vector, align 8
  %m_LeftOperation = getelementptr inbounds %"class.cl::sycl::vecdetail::SwizzleOp", %"class.cl::sycl::vecdetail::SwizzleOp"* %this1, i32 0, i32 1
  %m_RightOperation = getelementptr inbounds %"class.cl::sycl::vecdetail::SwizzleOp", %"class.cl::sycl::vecdetail::SwizzleOp"* %this1, i32 0, i32 2
  ret void
}

; CHECK-FOO1: define linkonce_odr dso_local spir_func dereferenceable(16) %"class.cl::sycl::vec.0" addrspace(4)* @new.[[FOO1]](%"class.cl::sycl::vec.0" addrspace(4)*, %"class.cl::sycl::vec.0" addrspace(4)* dereferenceable(16)) #2 align 2 {
; Function Attrs: noinline nounwind optnone
define linkonce_odr dso_local spir_func dereferenceable(16) %"class.cl::sycl::vec.0"* @_ZN2cl4sycl3vecIfLi3EEaSERKS2_(%"class.cl::sycl::vec.0"* %this, %"class.cl::sycl::vec.0"* dereferenceable(16) %Rhs) #2 comdat align 2 {
entry:
; CHECK-FOO1:  %new.[[THIS_ALLOC:.*]] = alloca %"class.cl::sycl::vec.0" addrspace(4)*
  %this.addr = alloca %"class.cl::sycl::vec.0"*, align 8
; CHECK-FOO1:  %new.[[RHS_ALLOC:.*]] = alloca %"class.cl::sycl::vec.0" addrspace(4)*
  %Rhs.addr = alloca %"class.cl::sycl::vec.0"*, align 8
; CHECK-FOO1:  store %"class.cl::sycl::vec.0" addrspace(4)* %{{.*}}, %"class.cl::sycl::vec.0" addrspace(4)** %new.[[THIS_ALLOC]], align 8
  store %"class.cl::sycl::vec.0"* %this, %"class.cl::sycl::vec.0"** %this.addr, align 8
; CHECK-FOO1:  store %"class.cl::sycl::vec.0" addrspace(4)* %{{.*}}, %"class.cl::sycl::vec.0" addrspace(4)** %new.[[RHS_ALLOC]], align 8
  store %"class.cl::sycl::vec.0"* %Rhs, %"class.cl::sycl::vec.0"** %Rhs.addr, align 8
; CHECK-FOO1:  %[[THIS_LOAD:.*]] = load %"class.cl::sycl::vec.0" addrspace(4)*, %"class.cl::sycl::vec.0" addrspace(4)** %new.[[THIS_ALLOC]]
  %this1 = load %"class.cl::sycl::vec.0"*, %"class.cl::sycl::vec.0"** %this.addr, align 8
; CHECK-FOO1:  %[[RHS_LOAD:.*]] = load %"class.cl::sycl::vec.0" addrspace(4)*, %"class.cl::sycl::vec.0" addrspace(4)** %new.[[RHS_ALLOC]]
  %0 = load %"class.cl::sycl::vec.0"*, %"class.cl::sycl::vec.0"** %Rhs.addr, align 8
; CHECK-FOO1:  %[[RHS_GEP:.*]] = getelementptr %"class.cl::sycl::vec.0", %"class.cl::sycl::vec.0" addrspace(4)* %[[RHS_LOAD]], i32 0, i32 0
  %m_Data = getelementptr inbounds %"class.cl::sycl::vec.0", %"class.cl::sycl::vec.0"* %0, i32 0, i32 0
; CHECK-FOO1:  %[[CAST_TO_VEC:.*]] = bitcast <3 x float> addrspace(4)* %[[RHS_GEP]] to <4 x float> addrspace(4)*
  %castToVec4 = bitcast <3 x float>* %m_Data to <4 x float>*
; CHECK-FOO1:  %[[VEC_LOAD:.*]] = load <4 x float>, <4 x float> addrspace(4)* %[[CAST_TO_VEC]]
  %loadVec4 = load <4 x float>, <4 x float>* %castToVec4, align 16
; CHECK-FOO1:  %[[EX_VEC:.*]] = shufflevector <4 x float> %[[VEC_LOAD]], <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
  %extractVec = shufflevector <4 x float> %loadVec4, <4 x float> undef, <3 x i32> <i32 0, i32 1, i32 2>
; CHECK-FOO1:  %[[THIS_GEP:.*]] = getelementptr %"class.cl::sycl::vec.0", %"class.cl::sycl::vec.0" addrspace(4)* %[[THIS_LOAD]], i32 0, i32 0
  %m_Data2 = getelementptr inbounds %"class.cl::sycl::vec.0", %"class.cl::sycl::vec.0"* %this1, i32 0, i32 0
; CHECK-FOO1:  %[[EX_VEC1:.*]] = shufflevector <3 x float> %[[EX_VEC]], <3 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
  %extractVec3 = shufflevector <3 x float> %extractVec, <3 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>
; CHECK-FOO1:  %[[GEP_CAST:.*]] = bitcast <3 x float> addrspace(4)* %[[THIS_GEP]] to <4 x float> addrspace(4)*
  %storetmp = bitcast <3 x float>* %m_Data2 to <4 x float>*
; CHECK-FOO1:  store <4 x float> %[[EX_VEC1]], <4 x float> addrspace(4)* %[[GEP_CAST]], align 16
  store <4 x float> %extractVec3, <4 x float>* %storetmp, align 16
; CHECK-FOO1;  ret %"class.cl::sycl::vec.0" addrspace(4)* %[[THIS_LOAD]]
  ret %"class.cl::sycl::vec.0"* %this1
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { noinline "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 8.0.0"}
