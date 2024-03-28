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

; CHECK-LABEL: @poison_constant
define void @poison_constant(double %arg0) {
  ; CHECK:  %[[POISON:.+]] = llvm.mlir.poison : f64
  ; CHECK:  llvm.fadd %[[POISON]], %{{.*}} : f64
  %1 = fadd double poison, %arg0
  ret void
}

; // -----

; CHECK-LABEL: @null_constant
define ptr @null_constant() {
  ; CHECK:  %[[NULL:[0-9]+]] = llvm.mlir.zero : !llvm.ptr
  ; CHECK:  llvm.return %[[NULL]] : !llvm.ptr
  ret ptr null
}

; // -----

@global = external global i32, align 8

; CHECK-LABEL: @gep_const_expr
define ptr @gep_const_expr() {
  ; CHECK-DAG:  %[[ADDR:[0-9]+]] = llvm.mlir.addressof @global : !llvm.ptr
  ; CHECK-DAG:  %[[IDX:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
  ; CHECK-DAG:  %[[GEP:[0-9]+]] = llvm.getelementptr %[[ADDR]][%[[IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr
  ; CHECK-DAG:  llvm.return %[[GEP]] : !llvm.ptr
  ret ptr getelementptr (i32, ptr @global, i32 2)
}

; // -----

@global = external global i32, align 8

; CHECK-LABEL: @const_expr_with_duplicate
define i64 @const_expr_with_duplicate() {
  ; CHECK-DAG:  %[[ADDR:[0-9]+]] = llvm.mlir.addressof @global : !llvm.ptr
  ; CHECK-DAG:  %[[IDX:[0-9]+]] = llvm.mlir.constant(7 : i32) : i32
  ; CHECK-DAG:  %[[GEP:[0-9]+]] = llvm.getelementptr %[[ADDR]][%[[IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr
  ; CHECK-DAG:  %[[DUP:[0-9]+]] = ptr.ptrtoint %[[GEP]] : !llvm.ptr to i64

  ; Verify the duplicate sub expression is converted only once.
  ; CHECK-DAG:  %[[SUM:[0-9]+]] = llvm.add %[[DUP]], %[[DUP]] : i64
  ; CHECK-DAG:  llvm.return %[[SUM]] : i64
  ret i64 add (i64 ptrtoint (ptr getelementptr (i32, ptr @global, i32 7) to i64),
               i64 ptrtoint (ptr getelementptr (i32, ptr @global, i32 7) to i64))
}

; // -----

@global = external global i32, align 8

; CHECK-LABEL: @const_expr_with_aggregate()
define i64 @const_expr_with_aggregate() {
  ; Compute the vector elements.
  ; CHECK-DAG:  %[[VAL1:[0-9]+]] = llvm.mlir.constant(33 : i64) : i64
  ; CHECK-DAG:  %[[ADDR:[0-9]+]] = llvm.mlir.addressof @global : !llvm.ptr
  ; CHECK-DAG:  %[[IDX1:[0-9]+]] = llvm.mlir.constant(7 : i32) : i32
  ; CHECK-DAG:  %[[GEP1:[0-9]+]] = llvm.getelementptr %[[ADDR]][%[[IDX1]]] : (!llvm.ptr, i32) -> !llvm.ptr
  ; CHECK-DAG:  %[[VAL2:[0-9]+]] = ptr.ptrtoint %[[GEP1]] : !llvm.ptr to i64

  ; Fill the vector.
  ; CHECK-DAG:  %[[VEC1:[0-9]+]] = llvm.mlir.undef : vector<2xi64>
  ; CHECK-DAG:  %[[IDX2:[0-9]+]] = llvm.mlir.constant(0 : i32) : i32
  ; CHECK-DAG:  %[[VEC2:[0-9]+]] = llvm.insertelement %[[VAL1]], %[[VEC1]][%[[IDX2]] : i32] : vector<2xi64>
  ; CHECK-DAG:  %[[IDX3:[0-9]+]] = llvm.mlir.constant(1 : i32) : i32
  ; CHECK-DAG:  %[[VEC3:[0-9]+]] = llvm.insertelement %[[VAL2]], %[[VEC2]][%[[IDX3]] : i32] : vector<2xi64>
  ; CHECK-DAG:  %[[IDX4:[0-9]+]] = llvm.mlir.constant(42 : i32) : i32

  ; Compute the extract index.
  ; CHECK-DAG:  %[[GEP2:[0-9]+]] = llvm.getelementptr %[[ADDR]][%[[IDX4]]] : (!llvm.ptr, i32) -> !llvm.ptr
  ; CHECK-DAG:  %[[IDX5:[0-9]+]] = ptr.ptrtoint %[[GEP2]] : !llvm.ptr to i64

  ; Extract the vector element.
  ; CHECK-DAG:  %[[ELEM:[0-9]+]] = llvm.extractelement %[[VEC3]][%[[IDX5]] : i64] : vector<2xi64>
  ; CHECK-DAG:  llvm.return %[[ELEM]] : i64
  ret i64 extractelement (
    <2 x i64> <i64 33, i64 ptrtoint (ptr getelementptr (i32, ptr @global, i32 7) to i64)>,
    i64 ptrtoint (ptr getelementptr (i32, ptr @global, i32 42) to i64))
}

; // -----

; Verify the function constant import.

; Calling a function that has not been defined yet.
; CHECK-LABEL: @function_address_before_def
define i32 @function_address_before_def() {
  %1 = alloca ptr
  ; CHECK:  %[[FUN:.*]] = llvm.mlir.addressof @callee : !llvm.ptr
  ; CHECK:  ptr.store %[[FUN]], %[[PTR:.*]] : !llvm.ptr, !llvm.ptr
  store ptr @callee, ptr %1
  ; CHECK:  %[[INDIR:.*]] = ptr.load %[[PTR]] : !llvm.ptr -> !llvm.ptr
  %2 = load ptr, ptr %1
  ; CHECK:  llvm.call %[[INDIR]]() : !llvm.ptr, () -> i32
  %3 = call i32 %2()
  ret i32 %3
}

define i32 @callee() {
  ret i32 42
}

; Calling a function that has been defined.
; CHECK-LABEL: @function_address_after_def
define i32 @function_address_after_def() {
  %1 = alloca ptr
  ; CHECK:  %[[FUN:.*]] = llvm.mlir.addressof @callee : !llvm.ptr
  ; CHECK:  ptr.store %[[FUN]], %[[PTR:.*]] : !llvm.ptr, !llvm.ptr
  store ptr @callee, ptr %1
  ; CHECK:  %[[INDIR:.*]] = ptr.load %[[PTR]] : !llvm.ptr -> !llvm.ptr
  %2 = load ptr, ptr %1
  ; CHECK:  llvm.call %[[INDIR]]() : !llvm.ptr, () -> i32
  %3 = call i32 %2()
  ret i32 %3
}

; // -----

; Verify the aggregate constant import.

; CHECK-DAG:  %[[C0:.+]] = llvm.mlir.constant(9 : i32) : i32
; CHECK-DAG:  %[[C1:.+]] = llvm.mlir.constant(4 : i8) : i8
; CHECK-DAG:  %[[C2:.+]] = llvm.mlir.constant(8 : i16) : i16
; CHECK-DAG:  %[[C3:.+]] = llvm.mlir.constant(7 : i32) : i32
; CHECK-DAG:  %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"simple_agg_type", (i32, i8, i16, i32)>
; CHECK-DAG:  %[[CHAIN0:.+]] = llvm.insertvalue %[[C0]], %[[ROOT]][0]
; CHECK-DAG:  %[[CHAIN1:.+]] = llvm.insertvalue %[[C1]], %[[CHAIN0]][1]
; CHECK-DAG:  %[[CHAIN2:.+]] = llvm.insertvalue %[[C2]], %[[CHAIN1]][2]
; CHECK-DAG:  %[[CHAIN3:.+]] = llvm.insertvalue %[[C3]], %[[CHAIN2]][3]
; CHECK-DAG:  llvm.return %[[CHAIN3]]
%simple_agg_type = type {i32, i8, i16, i32}
@simple_agg = global %simple_agg_type {i32 9, i8 4, i16 8, i32 7}

; CHECK-DAG:  %[[C1:.+]] = llvm.mlir.constant(1 : i32) : i32
; CHECK-DAG:  %[[C2:.+]] = llvm.mlir.constant(2 : i8) : i8
; CHECK-DAG:  %[[C3:.+]] = llvm.mlir.constant(3 : i16) : i16
; CHECK-DAG:  %[[C4:.+]] = llvm.mlir.constant(4 : i32) : i32
; CHECK-DAG:  %[[NESTED:.+]] = llvm.mlir.undef : !llvm.struct<"simple_agg_type", (i32, i8, i16, i32)>
; CHECK-DAG:  %[[CHAIN0:.+]] = llvm.insertvalue %[[C1]], %[[NESTED]][0]
; CHECK-DAG:  %[[CHAIN1:.+]] = llvm.insertvalue %[[C2]], %[[CHAIN0]][1]
; CHECK-DAG:  %[[CHAIN2:.+]] = llvm.insertvalue %[[C3]], %[[CHAIN1]][2]
; CHECK-DAG:  %[[CHAIN3:.+]] = llvm.insertvalue %[[C4]], %[[CHAIN2]][3]
; CHECK-DAG:  %[[NULL:.+]] = llvm.mlir.zero : !llvm.ptr
; CHECK-DAG:  %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"nested_agg_type", (struct<"simple_agg_type", (i32, i8, i16, i32)>, ptr)>
; CHECK-DAG:  %[[CHAIN4:.+]] = llvm.insertvalue %[[CHAIN3]], %[[ROOT]][0]
; CHECK-DAG:  %[[CHAIN5:.+]] = llvm.insertvalue %[[NULL]], %[[CHAIN4]][1]
; CHECK-DAG:  llvm.return %[[CHAIN5]]
%nested_agg_type = type {%simple_agg_type, ptr}
@nested_agg = global %nested_agg_type { %simple_agg_type{i32 1, i8 2, i16 3, i32 4}, ptr null }

; CHECK-DAG:  %[[NULL:.+]] = llvm.mlir.zero : !llvm.ptr
; CHECK-DAG:  %[[ROOT:.+]] = llvm.mlir.undef : !llvm.vec<2 x ptr>
; CHECK-DAG:  %[[P0:.+]] = llvm.mlir.constant(0 : i32) : i32
; CHECK-DAG:  %[[CHAIN0:.+]] = llvm.insertelement %[[NULL]], %[[ROOT]][%[[P0]] : i32] : !llvm.vec<2 x ptr>
; CHECK-DAG:  %[[P1:.+]] = llvm.mlir.constant(1 : i32) : i32
; CHECK-DAG:  %[[CHAIN1:.+]] = llvm.insertelement %[[NULL]], %[[CHAIN0]][%[[P1]] : i32] : !llvm.vec<2 x ptr>
; CHECK-DAG:  llvm.return %[[CHAIN1]] : !llvm.vec<2 x ptr>
@vector_agg = global <2 x ptr> <ptr null, ptr null>

; // -----

; Verfiy the import of subsequent constant expressions with duplicates.

@global = external global i32, align 8

; CHECK-LABEL: @const_exprs_with_duplicate
define i64 @const_exprs_with_duplicate() {
  ; CHECK: %[[ADDR:.+]] = llvm.mlir.addressof @global : !llvm.ptr
  ; CHECK: llvm.getelementptr %[[ADDR]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr
  %1 = add i64 1, ptrtoint (ptr getelementptr (i32, ptr @global, i32 7) to i64)

  ; Verify the address value is reused.
  ; CHECK: llvm.getelementptr %[[ADDR]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr
  %2 = add i64 %1, ptrtoint (ptr getelementptr (i32, ptr @global, i32 42) to i64)
  ret i64 %2
}

; // -----

; Verify the import of constant expressions with cyclic dependencies.

@cyclic = internal constant i64 add (i64 ptrtoint (ptr @cyclic to i64), i64 ptrtoint (ptr @cyclic to i64))

; CHECK-LABEL: @cyclic
; CHECK:  %[[ADDR:.+]] = llvm.mlir.addressof @cyclic
; CHECK:  %[[VAL0:.+]] = ptr.ptrtoint %[[ADDR]]
; CHECK:  %[[VAL1:.+]] = llvm.add %[[VAL0]], %[[VAL0]]
; CHECK:  llvm.return %[[VAL1]]
