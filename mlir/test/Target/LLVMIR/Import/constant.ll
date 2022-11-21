; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK-LABEL: @int_constants
define void @int_constants(i16 %arg0, i32 %arg1, i1 %arg2) {
  ; CHECK:  %[[C0:.+]] = llvm.mlir.constant(42 : i16) : i16
  ; CHECK:  %[[C1:.+]] = llvm.mlir.constant(7 : i32) : i32
  ; CHECK:  %[[C2:.+]] = llvm.mlir.constant(true) : i1

  ; CHECK:  llvm.add %[[C0]], %{{.*}} : i16
  %1 = add i16 42, %arg0
  ; CHECK:  llvm.add %[[C1]], %{{.*}} : i32
  %2 = add i32 7, %arg1
  ; CHECK:  llvm.or %[[C2]], %{{.*}} : i1
  %3 = or i1 1, %arg2
  ret void
}

; // -----

; CHECK-LABEL: @float_constants
define void @float_constants(half %arg0, bfloat %arg1, fp128 %arg2, x86_fp80 %arg3) {
  ; CHECK:  %[[C0:.+]] = llvm.mlir.constant(1.000000e+00 : f16) : f16
  ; CHECK:  %[[C1:.+]] = llvm.mlir.constant(1.000000e+00 : bf16) : bf16
  ; CHECK:  %[[C2:.+]] = llvm.mlir.constant(0.000000e+00 : f128) : f128
  ; CHECK:  %[[C3:.+]] = llvm.mlir.constant(7.000000e+00 : f80) : f80

  ; CHECK:  llvm.fadd %[[C0]], %{{.*}} : f16
  %1 = fadd half 1.0, %arg0
  ; CHECK:  llvm.fadd %[[C1]], %{{.*}} : bf16
  %2 = fadd bfloat 1.0, %arg1
  ; CHECK:  llvm.fadd %[[C2]], %{{.*}} : f128
  %3 = fadd fp128 0xL00000000000000000000000000000000, %arg2
  ; CHECK:  llvm.fadd %[[C3]], %{{.*}} : f80
  %4 = fadd x86_fp80 0xK4001E000000000000000, %arg3
  ret void
}

; // -----

; CHECK-LABEL: @undef_constant
define void @undef_constant(i32 %arg0) {
  ; CHECK:  %[[UNDEF:.+]] = llvm.mlir.undef : i32
  ; CHECK:  llvm.add %[[UNDEF]], %{{.*}} : i32
  %1 = add i32 undef, %arg0
  ret void
}

; // -----

; CHECK-LABEL: @null_constant
define i32* @null_constant() {
  ; CHECK:  %[[NULL:[0-9]+]] = llvm.mlir.null : !llvm.ptr<i32>
  ; CHECK:  llvm.return %[[NULL]] : !llvm.ptr<i32>
  ret i32* bitcast (double* null to i32*)
}

; // -----

@global = external global double, align 8

; CHECK-LABEL: @bitcast_const_expr
define i32* @bitcast_const_expr() {
  ; CHECK:  %[[VAL0:.*]] = llvm.mlir.addressof @global : !llvm.ptr<f64>
  ; CHECK:  %[[VAL1:.*]] = llvm.bitcast %[[VAL0]] : !llvm.ptr<f64> to !llvm.ptr<i32>
  ; CHECK:  llvm.return %[[VAL1]] : !llvm.ptr<i32>
  ret i32* bitcast (double* @global to i32*)
}

; // -----

@global = external global i32, align 8

; CHECK-LABEL: @gep_const_expr
define i32* @gep_const_expr() {
  ; CHECK:  %[[ADDR:[0-9]+]] = llvm.mlir.addressof @global : !llvm.ptr<i32>
  ; CHECK:  %[[IDX:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
  ; CHECK:  %[[GEP:[0-9]+]] = llvm.getelementptr %[[ADDR]][%[[IDX]]] : (!llvm.ptr<i32>, i32) -> !llvm.ptr<i32>
  ; CHECK:  llvm.return %[[GEP]] : !llvm.ptr<i32>
  ret i32* getelementptr (i32, i32* @global, i32 2)
}

; // -----

; Verify the function constant import.

; Calling a function that has not been defined yet.
; CHECK-LABEL: @function_address_before_def
define i32 @function_address_before_def() {
  %1 = alloca i32 ()*
  ; CHECK:  %[[FUN:.*]] = llvm.mlir.addressof @callee : !llvm.ptr<func<i32 ()>>
  ; CHECK:  llvm.store %[[FUN]], %[[PTR:.*]]
  store i32 ()* @callee, i32 ()** %1
  ; CHECK:  %[[INDIR:.*]] = llvm.load %[[PTR]]
  %2 = load i32 ()*, i32 ()** %1
  ; CHECK:  llvm.call %[[INDIR]]()
  %3 = call i32 %2()
  ret i32 %3
}

define i32 @callee() {
  ret i32 42
}

; Calling a function that has been defined.
; CHECK-LABEL: @function_address_after_def
define i32 @function_address_after_def() {
  %1 = alloca i32 ()*
  ; CHECK:  %[[FUN:.*]] = llvm.mlir.addressof @callee : !llvm.ptr<func<i32 ()>>
  ; CHECK:  llvm.store %[[FUN]], %[[PTR:.*]]
  store i32 ()* @callee, i32 ()** %1
  ; CHECK:  %[[INDIR:.*]] = llvm.load %[[PTR]]
  %2 = load i32 ()*, i32 ()** %1
  ; CHECK:  llvm.call %[[INDIR]]()
  %3 = call i32 %2()
  ret i32 %3
}

; // -----

; Verify the aggregate constant import.

; CHECK:  %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"simple_agg_type", (i32, i8, i16, i32)>
; CHECK:  %[[C0:.+]] = llvm.mlir.constant(9 : i32) : i32
; CHECK:  %[[CHAIN0:.+]] = llvm.insertvalue %[[C0]], %[[ROOT]][0]
; CHECK:  %[[C1:.+]] = llvm.mlir.constant(4 : i8) : i8
; CHECK:  %[[CHAIN1:.+]] = llvm.insertvalue %[[C1]], %[[CHAIN0]][1]
; CHECK:  %[[C2:.+]] = llvm.mlir.constant(8 : i16) : i16
; CHECK:  %[[CHAIN2:.+]] = llvm.insertvalue %[[C2]], %[[CHAIN1]][2]
; CHECK:  %[[C3:.+]] = llvm.mlir.constant(7 : i32) : i32
; CHECK:  %[[CHAIN3:.+]] = llvm.insertvalue %[[C3]], %[[CHAIN2]][3]
; CHECK:  llvm.return %[[CHAIN3]]
%simple_agg_type = type {i32, i8, i16, i32}
@simple_agg = global %simple_agg_type {i32 9, i8 4, i16 8, i32 7}

; CHECK:  %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"nested_agg_type", (struct<"simple_agg_type", (i32, i8, i16, i32)>, ptr<struct<"simple_agg_type", (i32, i8, i16, i32)>>)>
; CHECK:  %[[NESTED:.+]] = llvm.mlir.undef : !llvm.struct<"simple_agg_type", (i32, i8, i16, i32)>
; CHECK:  %[[C1:.+]] = llvm.mlir.constant(1 : i32) : i32
; CHECK:  %[[CHAIN0:.+]] = llvm.insertvalue %[[C1]], %[[NESTED]][0]
; CHECK:  %[[C2:.+]] = llvm.mlir.constant(2 : i8) : i8
; CHECK:  %[[CHAIN1:.+]] = llvm.insertvalue %[[C2]], %[[CHAIN0]][1]
; CHECK:  %[[C3:.+]] = llvm.mlir.constant(3 : i16) : i16
; CHECK:  %[[CHAIN2:.+]] = llvm.insertvalue %[[C3]], %[[CHAIN1]][2]
; CHECK:  %[[C4:.+]] = llvm.mlir.constant(4 : i32) : i32
; CHECK:  %[[CHAIN3:.+]] = llvm.insertvalue %[[C4]], %[[CHAIN2]][3]
; CHECK:  %[[CHAIN4:.+]] = llvm.insertvalue %[[CHAIN3]], %[[ROOT]][0]
; CHECK:  %[[NP:.+]] = llvm.mlir.null : !llvm.ptr<struct<"simple_agg_type", (i32, i8, i16, i32)>>
; CHECK:  %[[CHAIN5:.+]] = llvm.insertvalue %[[NP]], %[[CHAIN4]][1]
; CHECK:  llvm.return %[[CHAIN5]]
%nested_agg_type = type {%simple_agg_type, %simple_agg_type*}
@nested_agg = global %nested_agg_type { %simple_agg_type{i32 1, i8 2, i16 3, i32 4}, %simple_agg_type* null }

; CHECK:  %[[ROOT:.+]] = llvm.mlir.undef : !llvm.vec<2 x ptr<struct<"simple_agg_type", (i32, i8, i16, i32)>>>
; CHECK:  %[[C0:.+]] = llvm.mlir.null : !llvm.ptr<struct<"simple_agg_type", (i32, i8, i16, i32)>>
; CHECK:  %[[P0:.+]] = llvm.mlir.constant(0 : i32) : i32
; CHECK:  %[[CHAIN0:.+]] = llvm.insertelement %[[C0]], %[[ROOT]][%[[P0]] : i32] : !llvm.vec<2 x ptr<struct<"simple_agg_type", (i32, i8, i16, i32)>>>
; CHECK:  %[[C1:.+]] = llvm.mlir.null : !llvm.ptr<struct<"simple_agg_type", (i32, i8, i16, i32)>>
; CHECK:  %[[P1:.+]] = llvm.mlir.constant(1 : i32) : i32
; CHECK:  %[[CHAIN1:.+]] = llvm.insertelement %[[C1]], %[[CHAIN0]][%[[P1]] : i32] : !llvm.vec<2 x ptr<struct<"simple_agg_type", (i32, i8, i16, i32)>>>
; CHECK:  llvm.return %[[CHAIN1]] : !llvm.vec<2 x ptr<struct<"simple_agg_type", (i32, i8, i16, i32)>>>
@vector_agg = global <2 x %simple_agg_type*> <%simple_agg_type* null, %simple_agg_type* null>
