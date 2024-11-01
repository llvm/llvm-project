; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: @integer_arith
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG4:[a-zA-Z0-9]+]]
define void @integer_arith(i32 %arg1, i32 %arg2, i64 %arg3, i64 %arg4) {
  ; CHECK:  %[[C1:[0-9]+]] = llvm.mlir.constant(-7 : i32) : i32
  ; CHECK:  %[[C2:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
  ; CHECK:  llvm.add %[[ARG1]], %[[C1]] : i32
  %1 = add i32 %arg1, -7
  ; CHECK:  llvm.add %[[C2]], %[[ARG2]] : i32
  %2 = add i32 42, %arg2
  ; CHECK:  llvm.sub %[[ARG3]], %[[ARG4]] : i64
  %3 = sub i64 %arg3, %arg4
  ; CHECK:  llvm.mul %[[ARG1]], %[[ARG2]] : i32
  %4 = mul i32 %arg1, %arg2
  ; CHECK:  llvm.udiv %[[ARG3]], %[[ARG4]] : i64
  %5 = udiv i64 %arg3, %arg4
  ; CHECK:  llvm.sdiv %[[ARG1]], %[[ARG2]] : i32
  %6 = sdiv i32 %arg1, %arg2
  ; CHECK:  llvm.urem %[[ARG3]], %[[ARG4]] : i64
  %7 = urem i64 %arg3, %arg4
  ; CHECK:  llvm.srem %[[ARG1]], %[[ARG2]] : i32
  %8 = srem i32 %arg1, %arg2
  ; CHECK:  llvm.shl %[[ARG3]], %[[ARG4]] : i64
  %9 = shl i64 %arg3, %arg4
  ; CHECK:  llvm.lshr %[[ARG1]], %[[ARG2]] : i32
  %10 = lshr i32 %arg1, %arg2
  ; CHECK:  llvm.ashr %[[ARG3]], %[[ARG4]] : i64
  %11 = ashr i64 %arg3, %arg4
  ; CHECK:  llvm.and %[[ARG1]], %[[ARG2]] : i32
  %12 = and i32 %arg1, %arg2
  ; CHECK:  llvm.or %[[ARG3]], %[[ARG4]] : i64
  %13 = or i64 %arg3, %arg4
  ; CHECK:  llvm.xor %[[ARG1]], %[[ARG2]] : i32
  %14 = xor i32 %arg1, %arg2
  ret void
}

; // -----

; CHECK-LABEL: @integer_compare
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG4:[a-zA-Z0-9]+]]
define i1 @integer_compare(i32 %arg1, i32 %arg2, <4 x i64> %arg3, <4 x i64> %arg4) {
  ; CHECK:  llvm.icmp "eq" %[[ARG3]], %[[ARG4]] : vector<4xi64>
  %1 = icmp eq <4 x i64> %arg3, %arg4
  ; CHECK:  llvm.icmp "slt" %[[ARG1]], %[[ARG2]] : i32
  %2 = icmp slt i32 %arg1, %arg2
  ; CHECK:  llvm.icmp "sle" %[[ARG1]], %[[ARG2]] : i32
  %3 = icmp sle i32 %arg1, %arg2
  ; CHECK:  llvm.icmp "sgt" %[[ARG1]], %[[ARG2]] : i32
  %4 = icmp sgt i32 %arg1, %arg2
  ; CHECK:  llvm.icmp "sge" %[[ARG1]], %[[ARG2]] : i32
  %5 = icmp sge i32 %arg1, %arg2
  ; CHECK:  llvm.icmp "ult" %[[ARG1]], %[[ARG2]] : i32
  %6 = icmp ult i32 %arg1, %arg2
  ; CHECK:  llvm.icmp "ule" %[[ARG1]], %[[ARG2]] : i32
  %7 = icmp ule i32 %arg1, %arg2
  ; Verify scalar comparisons return a scalar boolean
  ; CHECK:  llvm.icmp "ugt" %[[ARG1]], %[[ARG2]] : i32
  %8 = icmp ugt i32 %arg1, %arg2
  ret i1 %8
}

; // -----

; CHECK-LABEL: @fp_arith
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG4:[a-zA-Z0-9]+]]
define void @fp_arith(float %arg1, float %arg2, double %arg3, double %arg4) {
  ; CHECK:  %[[C1:[0-9]+]] = llvm.mlir.constant(3.030000e+01 : f32) : f32
  ; CHECK:  %[[C2:[0-9]+]] = llvm.mlir.constant(3.030000e+01 : f64) : f64
  ; CHECK:  llvm.fadd %[[C1]], %[[ARG1]] : f32
  %1 = fadd float 0x403E4CCCC0000000, %arg1
  ; CHECK:  llvm.fadd %[[ARG1]], %[[ARG2]] : f32
  %2 = fadd float %arg1, %arg2
  ; CHECK:  llvm.fadd %[[C2]], %[[ARG3]] : f64
  %3 = fadd double 3.030000e+01, %arg3
  ; CHECK:  llvm.fsub %[[ARG1]], %[[ARG2]] : f32
  %4 = fsub float %arg1, %arg2
  ; CHECK:  llvm.fmul %[[ARG3]], %[[ARG4]] : f64
  %5 = fmul double %arg3, %arg4
  ; CHECK:  llvm.fdiv %[[ARG1]], %[[ARG2]] : f32
  %6 = fdiv float %arg1, %arg2
  ; CHECK:  llvm.frem %[[ARG3]], %[[ARG4]] : f64
  %7 = frem double %arg3, %arg4
  ; CHECK:  llvm.fneg %[[ARG1]] : f32
  %8 = fneg float %arg1
  ret void
}

; // -----

; CHECK-LABEL: @fp_compare
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG4:[a-zA-Z0-9]+]]
define <4 x i1> @fp_compare(float %arg1, float %arg2, <4 x double> %arg3, <4 x double> %arg4) {
  ; CHECK:  llvm.fcmp "_false" %[[ARG1]], %[[ARG2]] : f32
  %1 = fcmp false float %arg1, %arg2
  ; CHECK:  llvm.fcmp "oeq" %[[ARG1]], %[[ARG2]] : f32
  %2 = fcmp oeq float %arg1, %arg2
  ; CHECK:  llvm.fcmp "ogt" %[[ARG1]], %[[ARG2]] : f32
  %3 = fcmp ogt float %arg1, %arg2
  ; CHECK:  llvm.fcmp "oge" %[[ARG1]], %[[ARG2]] : f32
  %4 = fcmp oge float %arg1, %arg2
  ; CHECK:  llvm.fcmp "olt" %[[ARG1]], %[[ARG2]] : f32
  %5 = fcmp olt float %arg1, %arg2
  ; CHECK:  llvm.fcmp "ole" %[[ARG1]], %[[ARG2]] : f32
  %6 = fcmp ole float %arg1, %arg2
  ; CHECK:  llvm.fcmp "one" %[[ARG1]], %[[ARG2]] : f32
  %7 = fcmp one float %arg1, %arg2
  ; CHECK:  llvm.fcmp "ord" %[[ARG1]], %[[ARG2]] : f32
  %8 = fcmp ord float %arg1, %arg2
  ; CHECK:  llvm.fcmp "ueq" %[[ARG1]], %[[ARG2]] : f32
  %9 = fcmp ueq float %arg1, %arg2
  ; CHECK:  llvm.fcmp "ugt" %[[ARG1]], %[[ARG2]] : f32
  %10 = fcmp ugt float %arg1, %arg2
  ; CHECK:  llvm.fcmp "uge" %[[ARG1]], %[[ARG2]] : f32
  %11 = fcmp uge float %arg1, %arg2
  ; CHECK:  llvm.fcmp "ult" %[[ARG1]], %[[ARG2]] : f32
  %12 = fcmp ult float %arg1, %arg2
  ; CHECK:  llvm.fcmp "ule" %[[ARG1]], %[[ARG2]] : f32
  %13 = fcmp ule float %arg1, %arg2
  ; CHECK:  llvm.fcmp "une" %[[ARG1]], %[[ARG2]] : f32
  %14 = fcmp une float %arg1, %arg2
  ; CHECK:  llvm.fcmp "uno" %[[ARG1]], %[[ARG2]] : f32
  %15 = fcmp uno float %arg1, %arg2
  ; Verify vector comparisons return a vector of booleans
  ; CHECK:  llvm.fcmp "_true" %[[ARG3]], %[[ARG4]] : vector<4xf64>
  %16 = fcmp true <4 x double> %arg3, %arg4
  ret <4 x i1> %16
}

; // -----

; CHECK-LABEL: @fp_casts
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]
define void @fp_casts(float %arg1, double %arg2, i32 %arg3) {
  ; CHECK:  llvm.fptrunc %[[ARG2]] : f64 to f32
  ; CHECK:  llvm.fpext %[[ARG1]] : f32 to f64
  ; CHECK:  llvm.fptosi %[[ARG2]] : f64 to i16
  ; CHECK:  llvm.fptoui %[[ARG1]] : f32 to i32
  ; CHECK:  llvm.sitofp %[[ARG3]] : i32 to f32
  ; CHECK:  llvm.uitofp %[[ARG3]] : i32 to f64
  %1 = fptrunc double %arg2 to float
  %2 = fpext float %arg1 to double
  %3 = fptosi double %arg2 to i16
  %4 = fptoui float %arg1 to i32
  %5 = sitofp i32 %arg3 to float
  %6 = uitofp i32 %arg3 to double
  ret void
}

; // -----

; CHECK-LABEL: @integer_extension_and_truncation
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
define void @integer_extension_and_truncation(i32 %arg1) {
  ; CHECK:  llvm.sext %[[ARG1]] : i32 to i64
  ; CHECK:  llvm.zext %[[ARG1]] : i32 to i64
  ; CHECK:  llvm.trunc %[[ARG1]] : i32 to i16
  %1 = sext i32 %arg1 to i64
  %2 = zext i32 %arg1 to i64
  %3 = trunc i32 %arg1 to i16
  ret void
}

; // -----

; CHECK-LABEL: @pointer_casts
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
define ptr @pointer_casts(ptr %arg1, i64 %arg2) {
  ; CHECK:  %[[NULL:[0-9]+]] = llvm.mlir.null : !llvm.ptr
  ; CHECK:  llvm.ptrtoint %[[ARG1]] : !llvm.ptr to i64
  ; CHECK:  llvm.inttoptr %[[ARG2]] : i64 to !llvm.ptr
  ; CHECK:  llvm.bitcast %[[ARG1]] : !llvm.ptr to !llvm.ptr
  ; CHECK:  llvm.return %[[NULL]] : !llvm.ptr
  %1 = ptrtoint ptr %arg1 to i64
  %2 = inttoptr i64 %arg2 to ptr
  %3 = bitcast ptr %arg1 to ptr
  ret ptr null
}

; // -----

; CHECK-LABEL: @addrspace_casts
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
define ptr addrspace(2) @addrspace_casts(ptr addrspace(1) %arg1) {
  ; CHECK:  llvm.addrspacecast %[[ARG1]] : !llvm.ptr<1> to !llvm.ptr<2>
  ; CHECK:  llvm.return {{.*}} : !llvm.ptr<2>
  %1 = addrspacecast ptr addrspace(1) %arg1 to ptr addrspace(2)
  ret ptr addrspace(2) %1
}

; // -----

; CHECK-LABEL: @integer_arith
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG4:[a-zA-Z0-9]+]]
define void @integer_arith(i32 %arg1, i32 %arg2, i64 %arg3, i64 %arg4) {
  ; CHECK:  %[[C1:[0-9]+]] = llvm.mlir.constant(-7 : i32) : i32
  ; CHECK:  %[[C2:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
  ; CHECK:  llvm.add %[[ARG1]], %[[C1]] : i32
  ; CHECK:  llvm.add %[[C2]], %[[ARG2]] : i32
  ; CHECK:  llvm.sub %[[ARG3]], %[[ARG4]] : i64
  ; CHECK:  llvm.mul %[[ARG1]], %[[ARG2]] : i32
  ; CHECK:  llvm.udiv %[[ARG3]], %[[ARG4]] : i64
  ; CHECK:  llvm.sdiv %[[ARG1]], %[[ARG2]] : i32
  ; CHECK:  llvm.urem %[[ARG3]], %[[ARG4]] : i64
  ; CHECK:  llvm.srem %[[ARG1]], %[[ARG2]] : i32
  ; CHECK:  llvm.shl %[[ARG3]], %[[ARG4]] : i64
  ; CHECK:  llvm.lshr %[[ARG1]], %[[ARG2]] : i32
  ; CHECK:  llvm.ashr %[[ARG3]], %[[ARG4]] : i64
  ; CHECK:  llvm.and %[[ARG1]], %[[ARG2]] : i32
  ; CHECK:  llvm.or %[[ARG3]], %[[ARG4]] : i64
  ; CHECK:  llvm.xor %[[ARG1]], %[[ARG2]] : i32
  %1 = add i32 %arg1, -7
  %2 = add i32 42, %arg2
  %3 = sub i64 %arg3, %arg4
  %4 = mul i32 %arg1, %arg2
  %5 = udiv i64 %arg3, %arg4
  %6 = sdiv i32 %arg1, %arg2
  %7 = urem i64 %arg3, %arg4
  %8 = srem i32 %arg1, %arg2
  %9 = shl i64 %arg3, %arg4
  %10 = lshr i32 %arg1, %arg2
  %11 = ashr i64 %arg3, %arg4
  %12 = and i32 %arg1, %arg2
  %13 = or i64 %arg3, %arg4
  %14 = xor i32 %arg1, %arg2
  ret void
}

; // -----

; CHECK-LABEL: @extract_element
; CHECK-SAME:  %[[VEC:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[IDX:[a-zA-Z0-9]+]]
define half @extract_element(ptr %vec, i32 %idx) {
  ; CHECK:  %[[V1:.+]] = llvm.load %[[VEC]] {{.*}} : !llvm.ptr -> vector<4xf16>
  ; CHECK:  %[[V2:.+]] = llvm.extractelement %[[V1]][%[[IDX]] : i32] : vector<4xf16>
  ; CHECK:  llvm.return %[[V2]]
  %1 = load <4 x half>, ptr %vec
  %2 = extractelement <4 x half> %1, i32 %idx
  ret half %2
}

; // -----

; CHECK-LABEL: @insert_element
; CHECK-SAME:  %[[VEC:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[VAL:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[IDX:[a-zA-Z0-9]+]]
define <4 x half> @insert_element(ptr %vec, half %val, i32 %idx) {
  ; CHECK:  %[[V1:.+]] = llvm.load %[[VEC]] {{.*}} : !llvm.ptr -> vector<4xf16>
  ; CHECK:  %[[V2:.+]] = llvm.insertelement %[[VAL]], %[[V1]][%[[IDX]] : i32] : vector<4xf16>
  ; CHECK:  llvm.return %[[V2]]
  %1 = load <4 x half>, ptr %vec
  %2 = insertelement <4 x half> %1, half %val, i32 %idx
  ret <4 x half> %2
}

; // -----

; CHECK-LABEL: @insert_extract_value_struct
; CHECK-SAME:  %[[PTR:[a-zA-Z0-9]+]]
define float @insert_extract_value_struct({{i32},{float, double}}* %ptr) {
  ; CHECK:  %[[C0:.+]] = llvm.mlir.constant(2.000000e+00 : f64)
  ; CHECK:  %[[VT:.+]] = llvm.load %[[PTR]]
  %1 = load {{i32},{float, double}}, {{i32},{float, double}}* %ptr
  ; CHECK:  %[[EV:.+]] = llvm.extractvalue %[[VT]][1, 0] :
  ; CHECK-SAME: !llvm.struct<(struct<(i32)>, struct<(f32, f64)>)>
  %2 = extractvalue {{i32},{float, double}} %1, 1, 0
  ; CHECK:  %[[IV:.+]] = llvm.insertvalue %[[C0]], %[[VT]][1, 1] :
  ; CHECK-SAME: !llvm.struct<(struct<(i32)>, struct<(f32, f64)>)>
  %3 = insertvalue {{i32},{float, double}} %1, double 2.0, 1, 1
  ; CHECK:  llvm.store %[[IV]], %[[PTR]]
  store {{i32},{float, double}} %3, {{i32},{float, double}}* %ptr
  ; CHECK:  llvm.return %[[EV]]
  ret float %2
}

; // -----

; CHECK-LABEL: @insert_extract_value_array
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
define void @insert_extract_value_array([4 x [4 x i8]] %arg1) {
  ; CHECK:  %[[C0:.+]] = llvm.mlir.constant(0 : i8)
  ; CHECK:  llvm.insertvalue %[[C0]], %[[ARG1]][0, 0] : !llvm.array<4 x array<4 x i8>>
  %1 = insertvalue [4 x [4 x i8 ]] %arg1, i8 0, 0, 0
  ; CHECK:  llvm.extractvalue %[[ARG1]][1] : !llvm.array<4 x array<4 x i8>>
  %2 = extractvalue [4 x [4 x i8 ]] %arg1, 1
  ; CHECK:  llvm.extractvalue %[[ARG1]][0, 1] : !llvm.array<4 x array<4 x i8>>
  %3 = extractvalue [4 x [4 x i8 ]] %arg1, 0, 1
  ret void
}

; // -----

; CHECK-LABEL: @select
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[COND:[a-zA-Z0-9]+]]
define void @select(i32 %arg0, i32 %arg1, i1 %cond) {
  ; CHECK:  llvm.select %[[COND]], %[[ARG1]], %[[ARG2]] : i1, i32
  %1 = select i1 %cond, i32 %arg0, i32 %arg1
  ret void
}

; // -----

; CHECK-LABEL: func @shuffle_vec
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
define <4 x half> @shuffle_vec(<4 x half> %arg1, <4 x half> %arg2) {
  ; CHECK:  llvm.shufflevector %[[ARG1]], %[[ARG2]] [2, 3, -1, -1] : vector<4xf16>
  %1 = shufflevector <4 x half> %arg1, <4 x half> %arg2, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  ret <4 x half> %1
}

; // -----

; CHECK-LABEL: @alloca
; CHECK-SAME:  %[[SIZE:[a-zA-Z0-9]+]]
define ptr @alloca(i64 %size) {
  ; CHECK:  %[[C1:[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  ; CHECK:  llvm.alloca %[[C1]] x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  ; CHECK:  llvm.alloca %[[SIZE]] x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr
  ; CHECK:  llvm.alloca %[[SIZE]] x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr<3>
  %1 = alloca double
  %2 = alloca i32, i64 %size, align 8
  %3 = alloca i32, i64 %size, addrspace(3)
  ret ptr %1
}

; // -----

; CHECK-LABEL: @load_store
; CHECK-SAME:  %[[PTR:[a-zA-Z0-9]+]]
define void @load_store(ptr %ptr) {
  ; CHECK:  %[[V1:[0-9]+]] = llvm.load %[[PTR]] {alignment = 8 : i64} : !llvm.ptr -> f64
  ; CHECK:  %[[V2:[0-9]+]] = llvm.load volatile %[[PTR]] {alignment = 16 : i64, nontemporal} : !llvm.ptr -> f64
  %1 = load double, ptr %ptr
  %2 = load volatile double, ptr %ptr, align 16, !nontemporal !0

  ; CHECK:  llvm.store %[[V1]], %[[PTR]] {alignment = 8 : i64} : f64, !llvm.ptr
  ; CHECK:  llvm.store volatile %[[V2]], %[[PTR]] {alignment = 16 : i64, nontemporal} : f64, !llvm.ptr
  store double %1, ptr %ptr
  store volatile double %2, ptr %ptr, align 16, !nontemporal !0
  ret void
}

!0 = !{i32 1}

; // -----

; CHECK-LABEL: @atomic_load_store
; CHECK-SAME:  %[[PTR:[a-zA-Z0-9]+]]
define void @atomic_load_store(ptr %ptr) {
  ; CHECK:  %[[V1:[0-9]+]] = llvm.load %[[PTR]] atomic acquire {alignment = 8 : i64} : !llvm.ptr -> f64
  ; CHECK:  %[[V2:[0-9]+]] = llvm.load volatile %[[PTR]] atomic syncscope("singlethreaded") acquire {alignment = 16 : i64} : !llvm.ptr -> f64
  %1 = load atomic double, ptr %ptr acquire, align 8
  %2 = load atomic volatile double, ptr %ptr syncscope("singlethreaded") acquire, align 16

  ; CHECK:  llvm.store %[[V1]], %[[PTR]] atomic release {alignment = 8 : i64} : f64, !llvm.ptr
  ; CHECK:  llvm.store volatile %[[V2]], %[[PTR]] atomic syncscope("singlethreaded") release {alignment = 16 : i64} : f64, !llvm.ptr
  store atomic double %1, ptr %ptr release, align 8
  store atomic volatile double %2, ptr %ptr syncscope("singlethreaded") release, align 16
  ret void
}

; // -----

; CHECK-LABEL: @atomic_rmw
; CHECK-SAME:  %[[PTR1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[VAL1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[PTR2:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[VAL2:[a-zA-Z0-9]+]]
define void @atomic_rmw(ptr %ptr1, i32 %val1, ptr %ptr2, float %val2) {
  ; CHECK:  llvm.atomicrmw xchg %[[PTR1]], %[[VAL1]] acquire
  %1 = atomicrmw xchg ptr %ptr1, i32 %val1 acquire
  ; CHECK:  llvm.atomicrmw add %[[PTR1]], %[[VAL1]] release
  %2 = atomicrmw add ptr %ptr1, i32 %val1 release
  ; CHECK:  llvm.atomicrmw sub %[[PTR1]], %[[VAL1]] acq_rel
  %3 = atomicrmw sub ptr %ptr1, i32 %val1 acq_rel
  ; CHECK:  llvm.atomicrmw _and %[[PTR1]], %[[VAL1]] seq_cst
  %4 = atomicrmw and ptr %ptr1, i32 %val1 seq_cst
  ; CHECK:  llvm.atomicrmw nand %[[PTR1]], %[[VAL1]] acquire
  %5 = atomicrmw nand ptr %ptr1, i32 %val1 acquire
  ; CHECK:  llvm.atomicrmw _or %[[PTR1]], %[[VAL1]] acquire
  %6 = atomicrmw or ptr %ptr1, i32 %val1 acquire
  ; CHECK:  llvm.atomicrmw _xor %[[PTR1]], %[[VAL1]] acquire
  %7 = atomicrmw xor ptr %ptr1, i32 %val1 acquire
  ; CHECK:  llvm.atomicrmw max %[[PTR1]], %[[VAL1]] acquire
  %8 = atomicrmw max ptr %ptr1, i32 %val1 acquire
  ; CHECK:  llvm.atomicrmw min %[[PTR1]], %[[VAL1]] acquire
  %9 = atomicrmw min ptr %ptr1, i32 %val1 acquire
  ; CHECK:  llvm.atomicrmw umax %[[PTR1]], %[[VAL1]] acquire
  %10 = atomicrmw umax ptr %ptr1, i32 %val1 acquire
  ; CHECK:  llvm.atomicrmw umin %[[PTR1]], %[[VAL1]] acquire
  %11 = atomicrmw umin ptr %ptr1, i32 %val1 acquire
  ; CHECK:  llvm.atomicrmw fadd %[[PTR2]], %[[VAL2]] acquire
  %12 = atomicrmw fadd ptr %ptr2, float %val2 acquire
  ; CHECK:  llvm.atomicrmw fsub %[[PTR2]], %[[VAL2]] acquire
  %13 = atomicrmw fsub ptr %ptr2, float %val2 acquire
  ; CHECK:  llvm.atomicrmw fmax %[[PTR2]], %[[VAL2]] acquire
  %14 = atomicrmw fmax ptr %ptr2, float %val2 acquire
  ; CHECK:  llvm.atomicrmw fmin %[[PTR2]], %[[VAL2]] acquire
  %15 = atomicrmw fmin ptr %ptr2, float %val2 acquire
  ; CHECK:  llvm.atomicrmw uinc_wrap %[[PTR1]], %[[VAL1]] acquire
  %16 = atomicrmw uinc_wrap ptr %ptr1, i32 %val1 acquire
  ; CHECK:  llvm.atomicrmw udec_wrap %[[PTR1]], %[[VAL1]] acquire
  %17 = atomicrmw udec_wrap ptr %ptr1, i32 %val1 acquire

  ; CHECK:  llvm.atomicrmw volatile
  ; CHECK-SAME:  syncscope("singlethread")
  ; CHECK-SAME:  {alignment = 8 : i64}
  %18 = atomicrmw volatile udec_wrap ptr %ptr1, i32 %val1 syncscope("singlethread") acquire, align 8
  ret void
}

; // -----

; CHECK-LABEL: @atomic_cmpxchg
; CHECK-SAME:  %[[PTR1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[VAL1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[VAL2:[a-zA-Z0-9]+]]
define void @atomic_cmpxchg(ptr %ptr1, i32 %val1, i32 %val2) {
  ; CHECK:  llvm.cmpxchg %[[PTR1]], %[[VAL1]], %[[VAL2]] seq_cst seq_cst
  %1 = cmpxchg ptr %ptr1, i32 %val1, i32 %val2 seq_cst seq_cst
  ; CHECK:  llvm.cmpxchg %[[PTR1]], %[[VAL1]], %[[VAL2]] monotonic seq_cst
  %2 = cmpxchg ptr %ptr1, i32 %val1, i32 %val2 monotonic seq_cst

  ; CHECK:  llvm.cmpxchg weak volatile
  ; CHECK-SAME:  syncscope("singlethread")
  ; CHECK-SAME:  {alignment = 8 : i64}
  %3 = cmpxchg weak volatile ptr %ptr1, i32 %val1, i32 %val2 syncscope("singlethread") monotonic seq_cst, align 8
  ret void
}

; // -----

; CHECK: llvm.func @fn(i32) -> f32
declare float @fn(i32)

; CHECK-LABEL: @direct_call
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
define float @direct_call(i32 %arg1) {
  ; CHECK:  llvm.call @fn(%[[ARG1]])
  %1 = call float @fn(i32 %arg1)
  ret float %1
}

; // -----

; CHECK-LABEL: @indirect_call
; CHECK-SAME:  %[[PTR:[a-zA-Z0-9]+]]
define void @indirect_call(ptr addrspace(42) %fn) {
  ; CHECK:  %[[C0:[0-9]+]] = llvm.mlir.constant(0 : i16) : i16
  ; CHECK:  llvm.call %[[PTR]](%[[C0]]) : !llvm.ptr<42>, (i16) -> ()
  call addrspace(42) void %fn(i16 0)
  ret void
}

; // -----

; CHECK-LABEL: @gep_static_idx
; CHECK-SAME:  %[[PTR:[a-zA-Z0-9]+]]
define void @gep_static_idx(ptr %ptr) {
  ; CHECK: %[[IDX:.+]] = llvm.mlir.constant(7 : i32)
  ; CHECK: llvm.getelementptr inbounds %[[PTR]][%[[IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  %1 = getelementptr inbounds float, ptr %ptr, i32 7
  ret void
}

; // -----

; CHECK: @varargs(...)
declare void @varargs(...)

; CHECK-LABEL: @varargs_call
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
define void @varargs_call(i32 %0) {
  ; CHECK:  llvm.call @varargs(%[[ARG1]]) : (i32) -> ()
  call void (...) @varargs(i32 %0)
  ret void
}

; // -----

%sub_struct = type { i32, i8 }
%my_struct = type { %sub_struct, [4 x i32] }

; CHECK-LABEL: @gep_dynamic_idx
; CHECK-SAME:  %[[PTR:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[IDX:[a-zA-Z0-9]+]]
define void @gep_dynamic_idx(ptr %ptr, i32 %idx) {
  ; CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
  ; CHECK: llvm.getelementptr %[[PTR]][%[[C0]], 1, %[[IDX]]]{{.*}}"my_struct"
  %1 = getelementptr %my_struct, ptr %ptr, i32 0, i32 1, i32 %idx
  ret void
}

; // -----

; CHECK-LABEL: @freeze
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
define void @freeze(i32 %arg1) {
  ; CHECK:  %[[UNDEF:[0-9]+]] = llvm.mlir.undef : i64
  ; CHECK:  llvm.freeze %[[ARG1]] : i32
  ; CHECK:  llvm.freeze %[[UNDEF]] : i64
  %1 = freeze i32 %arg1
  %2 = freeze i64 undef
  ret void
}

; // -----

; CHECK-LABEL: @unreachable
define void @unreachable() {
  ; CHECK:  llvm.unreachable
  unreachable
}

; // -----

; CHECK-LABEL: @fence
define void @fence() {
  ; CHECK:  llvm.fence syncscope("agent") seq_cst
  ; CHECK:  llvm.fence release
  ; CHECK:  llvm.fence seq_cst
  fence syncscope("agent") seq_cst
  fence release
  fence syncscope("") seq_cst
  ret void
}
