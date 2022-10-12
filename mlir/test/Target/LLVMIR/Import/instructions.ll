; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: @integer_arith
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG4:[a-zA-Z0-9]+]]
define void @integer_arith(i32 %arg1, i32 %arg2, i64 %arg3, i64 %arg4) {
  ; CHECK-DAG:  %[[C1:[0-9]+]] = llvm.mlir.constant(-7 : i32) : i32
  ; CHECK-DAG:  %[[C2:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
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

; CHECK-LABEL: @fp_arith
; CHECK-SAME:  %[[ARG1:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG2:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG3:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[ARG4:[a-zA-Z0-9]+]]
define void @fp_arith(float %arg1, float %arg2, double %arg3, double %arg4) {
  ; CHECK:  %[[C1:[0-9]+]] = llvm.mlir.constant(3.030000e+01 : f64) : f64
  ; CHECK:  %[[C2:[0-9]+]] = llvm.mlir.constant(3.030000e+01 : f32) : f32
  ; CHECK:  llvm.fadd %[[C2]], %[[ARG1]] : f32
  ; CHECK:  llvm.fadd %[[ARG1]], %[[ARG2]] : f32
  ; CHECK:  llvm.fadd %[[C1]], %[[ARG3]] : f64
  ; CHECK:  llvm.fsub %[[ARG1]], %[[ARG2]] : f32
  ; CHECK:  llvm.fmul %[[ARG3]], %[[ARG4]] : f64
  ; CHECK:  llvm.fdiv %[[ARG1]], %[[ARG2]] : f32
  ; CHECK:  llvm.frem %[[ARG3]], %[[ARG4]] : f64
  ; CHECK:  llvm.fneg %[[ARG1]] : f32
  %1 = fadd float 0x403E4CCCC0000000, %arg1
  %2 = fadd float %arg1, %arg2
  %3 = fadd double 3.030000e+01, %arg3
  %4 = fsub float %arg1, %arg2
  %5 = fmul double %arg3, %arg4
  %6 = fdiv float %arg1, %arg2
  %7 = frem double %arg3, %arg4
  %8 = fneg float %arg1
  ret void
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
define i32* @pointer_casts(double* %arg1, i64 %arg2) {
  ; CHECK:  %[[NULL:[0-9]+]] = llvm.mlir.null : !llvm.ptr<i32>
  ; CHECK:  llvm.ptrtoint %[[ARG1]] : !llvm.ptr<f64> to i64
  ; CHECK:  llvm.inttoptr %[[ARG2]] : i64 to !llvm.ptr<i64>
  ; CHECK:  llvm.bitcast %[[ARG1]] : !llvm.ptr<f64> to !llvm.ptr<i32>
  ; CHECK:  llvm.return %[[NULL]] : !llvm.ptr<i32>
  %1 = ptrtoint double* %arg1 to i64
  %2 = inttoptr i64 %arg2 to i64*
  %3 = bitcast double* %arg1 to i32*
  ret i32* bitcast (double* null to i32*)
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
  ; CHECK-DAG:  %[[C1:[0-9]+]] = llvm.mlir.constant(-7 : i32) : i32
  ; CHECK-DAG:  %[[C2:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32
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
define half @extract_element(<4 x half>* %vec, i32 %idx) {
  ; CHECK:  %[[V1:.+]] = llvm.load %[[VEC]] : !llvm.ptr<vector<4xf16>>
  ; CHECK:  %[[V2:.+]] = llvm.extractelement %[[V1]][%[[IDX]] : i32] : vector<4xf16>
  ; CHECK:  llvm.return %[[V2]]
  %1 = load <4 x half>, <4 x half>* %vec
  %2 = extractelement <4 x half> %1, i32 %idx
  ret half %2
}

; // -----

; CHECK-LABEL: @insert_element
; CHECK-SAME:  %[[VEC:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[VAL:[a-zA-Z0-9]+]]
; CHECK-SAME:  %[[IDX:[a-zA-Z0-9]+]]
define <4 x half> @insert_element(<4 x half>* %vec, half %val, i32 %idx) {
  ; CHECK:  %[[V1:.+]] = llvm.load %[[VEC]] : !llvm.ptr<vector<4xf16>>
  ; CHECK:  %[[V2:.+]] = llvm.insertelement %[[VAL]], %[[V1]][%[[IDX]] : i32] : vector<4xf16>
  ; CHECK:  llvm.return %[[V2]]
  %1 = load <4 x half>, <4 x half>* %vec
  %2 = insertelement <4 x half> %1, half %val, i32 %idx
  ret <4 x half> %2
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

; CHECK-LABEL: @alloca
; CHECK-SAME:  %[[SIZE:[a-zA-Z0-9]+]]
define double* @alloca(i64 %size) {
  ; CHECK:  %[[C1:[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  ; CHECK:  llvm.alloca %[[C1]] x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr<f64>
  ; CHECK:  llvm.alloca %[[SIZE]] x i32 {alignment = 8 : i64} : (i64) -> !llvm.ptr<i32>
  ; CHECK:  llvm.alloca %[[SIZE]] x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr<i32, 3>
  %1 = alloca double
  %2 = alloca i32, i64 %size, align 8
  %3 = alloca i32, i64 %size, addrspace(3)
  ret double* %1
}

; // -----

; CHECK-LABEL: @load_store
; CHECK-SAME:  %[[PTR:[a-zA-Z0-9]+]]
define void @load_store(double* %ptr) {
  ; CHECK:  %[[V1:[0-9]+]] = llvm.load %[[PTR]] : !llvm.ptr<f64>
  ; CHECK:  llvm.store %[[V1]], %[[PTR]] : !llvm.ptr<f64>
  %1 = load double, double* %ptr
  store double %1, double* %ptr
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
