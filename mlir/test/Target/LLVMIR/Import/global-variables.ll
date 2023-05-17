; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

%sub_struct = type {}
%my_struct = type { %sub_struct, i64 }

; CHECK:  llvm.mlir.global external @global_struct
; CHECK-SAME:  {addr_space = 0 : i32, alignment = 8 : i64}
; CHECK-SAME:  !llvm.struct<"my_struct", (struct<"sub_struct", ()>, i64)>
@global_struct = external global %my_struct, align 8

; CHECK:  llvm.mlir.global external @global_float
; CHECK-SAME:  {addr_space = 0 : i32, alignment = 8 : i64} : f64
@global_float = external global double, align 8

; CHECK:  llvm.mlir.global internal constant @address_before
; CHECK:  = llvm.mlir.addressof @global_int : !llvm.ptr
@address_before = internal constant ptr @global_int

; CHECK:  llvm.mlir.global external @global_int
; CHECK-SAME:  {addr_space = 0 : i32, alignment = 8 : i64} : i32
@global_int = external global i32, align 8

; CHECK:  llvm.mlir.global internal constant @address_after
; CHECK:  = llvm.mlir.addressof @global_int : !llvm.ptr
@address_after = internal constant ptr @global_int

; CHECK:  llvm.mlir.global internal @global_string("hello world")
@global_string = internal global [11 x i8] c"hello world"

; CHECK:  llvm.mlir.global external @global_vector
; CHECK-SAME:  {addr_space = 0 : i32} : vector<8xi32>
@global_vector = external global <8 x i32>

; CHECK: llvm.mlir.global internal constant @global_gep_const_expr
; CHECK-SAME:  {addr_space = 0 : i32, dso_local} : !llvm.ptr {
; CHECK-DAG:  %[[ADDR:[0-9]+]] = llvm.mlir.addressof @global_int : !llvm.ptr
; CHECK-DAG:  %[[IDX:[0-9]+]] = llvm.mlir.constant(2 : i32) : i32
; CHECK-DAG:  %[[GEP:[0-9]+]] = llvm.getelementptr %[[ADDR]][%[[IDX]]] : (!llvm.ptr, i32) -> !llvm.ptr
; CHECK-DAG   llvm.return %[[GEP]] : !llvm.ptr
@global_gep_const_expr = internal constant ptr getelementptr (i32, ptr @global_int, i32 2)

; // -----

; alignment attribute.

; CHECK:  llvm.mlir.global private @global_int_align_32
; CHECK-SAME:  (42 : i64) {addr_space = 0 : i32, alignment = 32 : i64, dso_local} : i64
@global_int_align_32 = private global i64 42, align 32

; CHECK:  llvm.mlir.global private @global_int_align_64
; CHECK-SAME:  (42 : i64) {addr_space = 0 : i32, alignment = 64 : i64, dso_local} : i64
@global_int_align_64 = private global i64 42, align 64

; // -----

; dso_local attribute.

%sub_struct = type {}
%my_struct = type { %sub_struct, i64 }

; CHECK:  llvm.mlir.global external @dso_local_var
; CHECK-SAME:  {addr_space = 0 : i32, dso_local} : !llvm.struct<"my_struct", (struct<"sub_struct", ()>, i64)>
@dso_local_var = external dso_local global %my_struct

; // -----

; thread_local attribute.

%sub_struct = type {}
%my_struct = type { %sub_struct, i64 }

; CHECK:  llvm.mlir.global external thread_local @thread_local_var
; CHECK-SAME:  {addr_space = 0 : i32} : !llvm.struct<"my_struct", (struct<"sub_struct", ()>, i64)>
@thread_local_var = external thread_local global %my_struct

; // -----

; addr_space attribute.

; CHECK:  llvm.mlir.global external @addr_space_var
; CHECK-SAME:  (0 : i32) {addr_space = 6 : i32} : i32
@addr_space_var = addrspace(6) global i32 0

; // -----

; Linkage attributes.

; CHECK:  llvm.mlir.global private @private
; CHECK-SAME:  (42 : i32) {addr_space = 0 : i32, dso_local} : i32
@private = private global i32 42

; CHECK:  llvm.mlir.global internal @internal
; CHECK-SAME:  (42 : i32) {addr_space = 0 : i32, dso_local} : i32
@internal = internal global i32 42

; CHECK:  llvm.mlir.global available_externally @available_externally
; CHECK-SAME:  (42 : i32) {addr_space = 0 : i32}  : i32
@available_externally = available_externally global i32 42

; CHECK:  llvm.mlir.global linkonce @linkonce
; CHECK-SAME:  (42 : i32) {addr_space = 0 : i32} : i32
@linkonce = linkonce global i32 42

; CHECK:  llvm.mlir.global weak @weak
; CHECK-SAME:  (42 : i32) {addr_space = 0 : i32} : i32
@weak = weak global i32 42

; CHECK:  llvm.mlir.global common @common
; CHECK-SAME:  (0 : i32) {addr_space = 0 : i32} : i32
@common = common global i32 zeroinitializer

; CHECK:  llvm.mlir.global appending @appending
; CHECK-SAME:  (dense<[0, 1]> : tensor<2xi32>) {addr_space = 0 : i32} : !llvm.array<2 x i32>
@appending = appending global [2 x i32] [i32 0, i32 1]

; CHECK:  llvm.mlir.global extern_weak @extern_weak
; CHECK-SAME:  {addr_space = 0 : i32} : i32
@extern_weak = extern_weak global i32

; CHECK:  llvm.mlir.global linkonce_odr @linkonce_odr
; CHECK-SAME:  (42 : i32) {addr_space = 0 : i32} : i32
@linkonce_odr = linkonce_odr global i32 42

; CHECK:  llvm.mlir.global weak_odr @weak_odr
; CHECK-SAME:  (42 : i32) {addr_space = 0 : i32} : i32
@weak_odr = weak_odr global i32 42

; CHECK:  llvm.mlir.global external @external
; CHECK-SAME:  {addr_space = 0 : i32} : i32
@external = external global i32

; // -----

; local_unnamed_addr and unnamed_addr attributes.

; CHECK:  llvm.mlir.global private constant @no_unnamed_addr
; CHECK-SAME:  (42 : i64) {addr_space = 0 : i32, dso_local} : i64
@no_unnamed_addr = private constant i64 42

; CHECK:  llvm.mlir.global private local_unnamed_addr constant @local_unnamed_addr
; CHECK-SAME:  (42 : i64) {addr_space = 0 : i32, dso_local} : i64
@local_unnamed_addr = private local_unnamed_addr constant i64 42

; CHECK:  llvm.mlir.global private unnamed_addr constant @unnamed_addr
; CHECK-SAME:  (42 : i64) {addr_space = 0 : i32, dso_local} : i64
@unnamed_addr = private unnamed_addr constant i64 42

; // -----

; section attribute.

; CHECK:  llvm.mlir.global internal constant @sectionvar("hello world")
; CHECK-SAME:  {addr_space = 0 : i32, dso_local, section = ".mysection"}
@sectionvar = internal constant [11 x i8] c"hello world", section ".mysection"

; // -----

; Sequential constants.

; CHECK:  llvm.mlir.global internal constant @vector_constant
; CHECK-SAME:  (dense<[1, 2]> : vector<2xi32>)
; CHECK-SAME:  {addr_space = 0 : i32, dso_local} : vector<2xi32>
@vector_constant = internal constant <2 x i32> <i32 1, i32 2>

; CHECK:  llvm.mlir.global internal constant @array_constant
; CHECK-SAME:  (dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>)
; CHECK-SAME:  {addr_space = 0 : i32, dso_local} : !llvm.array<2 x f32>
@array_constant = internal constant [2 x float] [float 1., float 2.]

; CHECK: llvm.mlir.global internal constant @nested_array_constant
; CHECK-SAME-LITERAL:  (dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>)
; CHECK-SAME-LITERAL:  {addr_space = 0 : i32, dso_local} : !llvm.array<2 x array<2 x i32>>
@nested_array_constant = internal constant [2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]]

; CHECK: llvm.mlir.global internal constant @nested_array_constant3
; CHECK-SAME-LITERAL:  (dense<[[[1, 2], [3, 4]]]> : tensor<1x2x2xi32>)
; CHECK-SAME-LITERAL:  {addr_space = 0 : i32, dso_local} : !llvm.array<1 x array<2 x array<2 x i32>>>
@nested_array_constant3 = internal constant [1 x [2 x [2 x i32]]] [[2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]]]

; CHECK: llvm.mlir.global internal constant @nested_array_vector
; CHECK-SAME-LITERAL:  (dense<[[[1, 2], [3, 4]]]> : vector<1x2x2xi32>)
; CHECK-SAME-LITERAL:  {addr_space = 0 : i32, dso_local} : !llvm.array<1 x array<2 x vector<2xi32>>>
@nested_array_vector = internal constant [1 x [2 x <2 x i32>]] [[2 x <2 x i32>] [<2 x i32> <i32 1, i32 2>, <2 x i32> <i32 3, i32 4>]]

; CHECK:  llvm.mlir.global internal constant @vector_constant_zero
; CHECK-SAME:  (dense<0> : vector<2xi24>)
; CHECK-SAME:  {addr_space = 0 : i32, dso_local} : vector<2xi24>
@vector_constant_zero = internal constant <2 x i24> zeroinitializer

; CHECK:  llvm.mlir.global internal constant @array_constant_zero
; CHECK-SAME:  (dense<0.000000e+00> : tensor<2xbf16>)
; CHECK-SAME:  {addr_space = 0 : i32, dso_local} : !llvm.array<2 x bf16>
@array_constant_zero = internal constant [2 x bfloat] zeroinitializer

; CHECK: llvm.mlir.global internal constant @nested_array_constant3_zero
; CHECK-SAME:  (dense<0> : tensor<1x2x2xi32>)
; CHECK-SAME:  {addr_space = 0 : i32, dso_local} : !llvm.array<1 x array<2 x array<2 x i32>>>
@nested_array_constant3_zero = internal constant [1 x [2 x [2 x i32]]] zeroinitializer

; CHECK: llvm.mlir.global internal constant @nested_array_vector_zero
; CHECK-SAME:  (dense<0> : vector<1x2x2xi32>)
; CHECK-SAME:  {addr_space = 0 : i32, dso_local} : !llvm.array<1 x array<2 x vector<2xi32>>>
@nested_array_vector_zero = internal constant [1 x [2 x <2 x i32>]] zeroinitializer

; CHECK: llvm.mlir.global internal constant @nested_bool_array_constant
; CHECK-SAME-LITERAL:  (dense<[[true, false]]> : tensor<1x2xi1>)
; CHECK-SAME-LITERAL:  {addr_space = 0 : i32, dso_local} : !llvm.array<1 x array<2 x i1>>
@nested_bool_array_constant = internal constant [1 x [2 x i1]] [[2 x i1] [i1 1, i1 0]]

; CHECK: llvm.mlir.global internal constant @quad_float_constant
; CHECK-SAME:  dense<[
; CHECK-SAME:    529.340000000000031832314562052488327
; CHECK-SAME:    529.340000000001850821718107908964157
; CHECK-SAME:  ]> : vector<2xf128>)
; CHECK-SAME:  {addr_space = 0 : i32, dso_local} : vector<2xf128>
@quad_float_constant = internal constant <2 x fp128> <fp128 0xLF000000000000000400808AB851EB851, fp128 0xLF000000000000000400808AB851EB852>

; CHECK: llvm.mlir.global internal constant @quad_float_splat_constant
; CHECK-SAME:  dense<529.340000000000031832314562052488327> : vector<2xf128>)
; CHECK-SAME:  {addr_space = 0 : i32, dso_local} : vector<2xf128>
@quad_float_splat_constant = internal constant <2 x fp128> <fp128 0xLF000000000000000400808AB851EB851, fp128 0xLF000000000000000400808AB851EB851>

; // -----

; CHECK: llvm.mlir.global_ctors {ctors = [@foo, @bar], priorities = [0 : i32, 42 : i32]}
; CHECK: llvm.mlir.global_dtors {dtors = [@foo], priorities = [0 : i32]}
@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @foo, ptr null }, { i32, ptr, ptr } { i32 42, ptr @bar, ptr null }]
@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @foo, ptr null }]

define void @foo() {
  ret void
}

define void @bar() {
  ret void
}

; // -----

; Visibility attribute.

; CHECK: llvm.mlir.global external hidden constant @hidden("string")
@hidden = hidden constant [6 x i8] c"string"

; CHECK: llvm.mlir.global external protected constant @protected(42 : i64)
@protected = protected constant i64 42
