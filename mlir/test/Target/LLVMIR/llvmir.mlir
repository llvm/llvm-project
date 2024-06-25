// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK: @global_aligned32 = private global i64 42, align 32
"llvm.mlir.global"() ({}) {sym_name = "global_aligned32", global_type = i64, value = 42 : i64, linkage = #llvm.linkage<private>, alignment = 32} : () -> ()

// CHECK: @global_aligned64 = private global i64 42, align 64
llvm.mlir.global private @global_aligned64(42 : i64) {alignment = 64 : i64} : i64

// CHECK: @global_aligned64_native = private global i64 42, align 64
llvm.mlir.global private @global_aligned64_native(42 : i64) { alignment = 64 } : i64

// CHECK: @i32_global = internal global i32 42
llvm.mlir.global internal @i32_global(42: i32) : i32

// CHECK: @i32_const = internal constant i53 52
llvm.mlir.global internal constant @i32_const(52: i53) : i53

// CHECK: @int_global_array = internal global [3 x i32] [i32 62, i32 62, i32 62]
llvm.mlir.global internal @int_global_array(dense<62> : vector<3xi32>) : !llvm.array<3 x i32>

// CHECK: @int_global_array_zero_elements = internal constant [3 x [0 x [4 x float]]] zeroinitializer
llvm.mlir.global internal constant @int_global_array_zero_elements(dense<> : tensor<3x0x4xf32>) : !llvm.array<3 x array<0 x array<4 x f32>>>

// CHECK: @int_global_array_zero_elements_1d = internal constant [0 x float] zeroinitializer
llvm.mlir.global internal constant @int_global_array_zero_elements_1d(dense<> : tensor<0xf32>) : !llvm.array<0 x f32>

// CHECK: @i32_global_addr_space = internal addrspace(7) global i32 62
llvm.mlir.global internal @i32_global_addr_space(62: i32) {addr_space = 7 : i32} : i32

// CHECK: @float_global = internal global float 0.000000e+00
llvm.mlir.global internal @float_global(0.0: f32) : f32

// CHECK: @float_global_array = internal global [1 x float] [float -5.000000e+00]
llvm.mlir.global internal @float_global_array(dense<[-5.0]> : vector<1xf32>) : !llvm.array<1 x f32>

// CHECK: @string_const = internal constant [6 x i8] c"foobar"
llvm.mlir.global internal constant @string_const("foobar") : !llvm.array<6 x i8>

// CHECK: @int_global_undef = internal global i64 undef
llvm.mlir.global internal @int_global_undef() : i64

// CHECK: @f8E4M3FN_global_as_i8 = internal global i8 60
llvm.mlir.global internal @f8E4M3FN_global_as_i8(1.5 : f8E4M3FN) : i8

// CHECK: @f8E5M2_global_as_i8 = internal global i8 62
llvm.mlir.global internal @f8E5M2_global_as_i8(1.5 : f8E5M2) : i8

// CHECK: @f8E4M3FNUZ_global_as_i8 = internal global i8 68
llvm.mlir.global internal @f8E4M3FNUZ_global_as_i8(1.5 : f8E4M3FNUZ) : i8

// CHECK: @f8E5M2FNUZ_global_as_i8 = internal global i8 66
llvm.mlir.global internal @f8E5M2FNUZ_global_as_i8(1.5 : f8E5M2FNUZ) : i8

// CHECK: @f8E4M3B11FNUZ_global_as_i8 = internal global i8 92
llvm.mlir.global internal @f8E4M3B11FNUZ_global_as_i8(1.5 : f8E4M3B11FNUZ) : i8

// CHECK: @bf16_global_as_i16 = internal global i16 16320
llvm.mlir.global internal @bf16_global_as_i16(1.5 : bf16) : i16

// CHECK: @explicit_undef = global i32 undef
llvm.mlir.global external @explicit_undef() : i32 {
  %0 = llvm.mlir.undef : i32
  llvm.return %0 : i32
}

// CHECK: @int_gep = internal constant ptr getelementptr (i32, ptr @i32_global, i32 2)
llvm.mlir.global internal constant @int_gep() : !llvm.ptr {
  %addr = llvm.mlir.addressof @i32_global : !llvm.ptr
  %_c0 = llvm.mlir.constant(2: i32) : i32
  %gepinit = llvm.getelementptr %addr[%_c0] : (!llvm.ptr, i32) -> !llvm.ptr, i32
  llvm.return %gepinit : !llvm.ptr
}

// CHECK{LITERAL}: @dense_float_vector = internal global <3 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>
llvm.mlir.global internal @dense_float_vector(dense<[1.0, 2.0, 3.0]> : vector<3xf32>) : vector<3xf32>

// CHECK{LITERAL}: @splat_float_vector = internal global <3 x float> <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
llvm.mlir.global internal @splat_float_vector(dense<42.0> : vector<3xf32>) : vector<3xf32>

// CHECK{LITERAL}: @dense_double_vector = internal global <3 x double> <double 1.000000e+00, double 2.000000e+00, double 3.000000e+00>
llvm.mlir.global internal @dense_double_vector(dense<[1.0, 2.0, 3.0]> : vector<3xf64>) : vector<3xf64>

// CHECK{LITERAL}: @splat_double_vector = internal global <3 x double> <double 4.200000e+01, double 4.200000e+01, double 4.200000e+01>
llvm.mlir.global internal @splat_double_vector(dense<42.0> : vector<3xf64>) : vector<3xf64>

// CHECK{LITERAL}: @dense_i64_vector = internal global <3 x i64> <i64 1, i64 2, i64 3>
llvm.mlir.global internal @dense_i64_vector(dense<[1, 2, 3]> : vector<3xi64>) : vector<3xi64>

// CHECK{LITERAL}: @splat_i64_vector = internal global <3 x i64> <i64 42, i64 42, i64 42>
llvm.mlir.global internal @splat_i64_vector(dense<42> : vector<3xi64>) : vector<3xi64>

// CHECK{LITERAL}: @dense_float_vector_2d = internal global [2 x <2 x float>] [<2 x float> <float 1.000000e+00, float 2.000000e+00>, <2 x float> <float 3.000000e+00, float 4.000000e+00>]
llvm.mlir.global internal @dense_float_vector_2d(dense<[[1.0, 2.0], [3.0, 4.0]]> : vector<2x2xf32>) : !llvm.array<2 x vector<2xf32>>

// CHECK{LITERAL}: @splat_float_vector_2d = internal global [2 x <2 x float>] [<2 x float> <float 4.200000e+01, float 4.200000e+01>, <2 x float> <float 4.200000e+01, float 4.200000e+01>]
llvm.mlir.global internal @splat_float_vector_2d(dense<42.0> : vector<2x2xf32>) : !llvm.array<2 x vector<2xf32>>

// CHECK{LITERAL}: @dense_float_vector_3d = internal global [2 x [2 x <2 x float>]] [[2 x <2 x float>] [<2 x float> <float 1.000000e+00, float 2.000000e+00>, <2 x float> <float 3.000000e+00, float 4.000000e+00>], [2 x <2 x float>] [<2 x float> <float 5.000000e+00, float 6.000000e+00>, <2 x float> <float 7.000000e+00, float 8.000000e+00>]]
llvm.mlir.global internal @dense_float_vector_3d(dense<[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]> : vector<2x2x2xf32>) : !llvm.array<2 x !llvm.array<2 x vector<2xf32>>>

// CHECK{LITERAL}: @splat_float_vector_3d = internal global [2 x [2 x <2 x float>]] [[2 x <2 x float>] [<2 x float> <float 4.200000e+01, float 4.200000e+01>, <2 x float> <float 4.200000e+01, float 4.200000e+01>], [2 x <2 x float>] [<2 x float> <float 4.200000e+01, float 4.200000e+01>, <2 x float> <float 4.200000e+01, float 4.200000e+01>]]
llvm.mlir.global internal @splat_float_vector_3d(dense<42.0> : vector<2x2x2xf32>) : !llvm.array<2 x !llvm.array<2 x vector<2xf32>>>

//
// Linkage attribute.
//

// CHECK: @private = private global i32 42
llvm.mlir.global private @private(42 : i32) : i32
// CHECK: @internal = internal global i32 42
llvm.mlir.global internal @internal(42 : i32) : i32
// CHECK: @available_externally = available_externally global i32 42
llvm.mlir.global available_externally @available_externally(42 : i32) : i32
// CHECK: @linkonce = linkonce global i32 42
llvm.mlir.global linkonce @linkonce(42 : i32) : i32
// CHECK: @weak = weak global i32 42
llvm.mlir.global weak @weak(42 : i32) : i32
// CHECK: @common = common global i32 0
llvm.mlir.global common @common(0 : i32) : i32
// CHECK: @appending = appending global [3 x i32] [i32 1, i32 2, i32 3]
llvm.mlir.global appending @appending(dense<[1,2,3]> : tensor<3xi32>) : !llvm.array<3 x i32>
// CHECK: @extern_weak = extern_weak global i32
llvm.mlir.global extern_weak @extern_weak() : i32
// CHECK: @linkonce_odr = linkonce_odr global i32 42
llvm.mlir.global linkonce_odr @linkonce_odr(42 : i32) : i32
// CHECK: @weak_odr = weak_odr global i32 42
llvm.mlir.global weak_odr @weak_odr(42 : i32) : i32
// CHECK: @external = external global i32
llvm.mlir.global external @external() : i32


//
// Visibility attribute.
//

// CHECK: @hidden = hidden constant [6 x i8] c"string"
llvm.mlir.global external hidden constant @hidden("string")

// CHECK: @protected = protected constant i64 42
llvm.mlir.global external protected constant @protected(42 : i64) : i64

//
// UnnamedAddr attribute.
//

// CHECK: @no_unnamed_addr = private constant i64 42
llvm.mlir.global private constant @no_unnamed_addr(42 : i64) : i64
// CHECK: @local_unnamed_addr = private local_unnamed_addr constant i64 42
llvm.mlir.global private local_unnamed_addr constant @local_unnamed_addr(42 : i64) : i64
// CHECK: @unnamed_addr = private unnamed_addr constant i64 42
llvm.mlir.global private unnamed_addr constant @unnamed_addr(42 : i64) : i64

//
// dso_local attribute.
//

llvm.mlir.global @has_dso_local(42 : i64) {dso_local} : i64
// CHECK: @has_dso_local = dso_local global i64 42

//
// thr_local attribute.
//

llvm.mlir.global thread_local @has_thr_local(42 : i64) : i64
// CHECK: @has_thr_local = thread_local global i64 42

//
// Section attribute.
//

// CHECK: @sectionvar = internal constant [10 x i8] c"teststring", section ".mysection"
llvm.mlir.global internal constant @sectionvar("teststring")  {section = ".mysection"}: !llvm.array<10 x i8>

//
// Declarations of the allocation functions to be linked against. These are
// inserted before other functions in the module.
//

// CHECK: declare ptr @malloc(i64)
llvm.func @malloc(i64) -> !llvm.ptr
// CHECK: declare void @free(ptr)
llvm.func @free(!llvm.ptr)

//
// Basic functionality: function and block conversion, function calls,
// phi nodes, scalar type conversion, arithmetic operations.
//

// CHECK-LABEL: define void @empty()
// CHECK-NEXT:    ret void
// CHECK-NEXT:  }
llvm.func @empty() {
  llvm.return
}

// CHECK-LABEL: @global_refs
llvm.func @global_refs() {
  // Check load from globals.
  // CHECK: load i32, ptr @i32_global
  %0 = llvm.mlir.addressof @i32_global : !llvm.ptr
  %1 = llvm.load %0 : !llvm.ptr -> i32

  // Check the contracted form of load from array constants.
  // CHECK: load i8, ptr @string_const
  %2 = llvm.mlir.addressof @string_const : !llvm.ptr
  %c0 = llvm.mlir.constant(0 : index) : i64
  %3 = llvm.getelementptr %2[%c0, %c0] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<6 x i8>
  %4 = llvm.load %3 : !llvm.ptr -> i8

  llvm.return
}

// CHECK-LABEL: declare void @body(i64)
llvm.func @body(i64)


// CHECK-LABEL: define void @simple_loop()
llvm.func @simple_loop() {
// CHECK: br label %[[SIMPLE_bb1:[0-9]+]]
  llvm.br ^bb1

// Constants are inlined in LLVM rather than a separate instruction.
// CHECK: [[SIMPLE_bb1]]:
// CHECK-NEXT: br label %[[SIMPLE_bb2:[0-9]+]]
^bb1:   // pred: ^bb0
  %0 = llvm.mlir.constant(1 : index) : i64
  %1 = llvm.mlir.constant(42 : index) : i64
  llvm.br ^bb2(%0 : i64)

// CHECK: [[SIMPLE_bb2]]:
// CHECK-NEXT:   %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %[[SIMPLE_bb3:[0-9]+]] ], [ 1, %[[SIMPLE_bb1]] ]
// CHECK-NEXT:   %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 42
// CHECK-NEXT:   br i1 %{{[0-9]+}}, label %[[SIMPLE_bb3]], label %[[SIMPLE_bb4:[0-9]+]]
^bb2(%2: i64): // 2 preds: ^bb1, ^bb3
  %3 = llvm.icmp "slt" %2, %1 : i64
  llvm.cond_br %3, ^bb3, ^bb4

// CHECK: [[SIMPLE_bb3]]:
// CHECK-NEXT:   call void @body(i64 %{{[0-9]+}})
// CHECK-NEXT:   %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
// CHECK-NEXT:   br label %[[SIMPLE_bb2]]
^bb3:   // pred: ^bb2
  llvm.call @body(%2) : (i64) -> ()
  %4 = llvm.mlir.constant(1 : index) : i64
  %5 = llvm.add %2, %4 : i64
  llvm.br ^bb2(%5 : i64)

// CHECK: [[SIMPLE_bb4]]:
// CHECK-NEXT:    ret void
^bb4:   // pred: ^bb2
  llvm.return
}

// CHECK-LABEL: define void @simple_caller()
// CHECK-NEXT:   call void @simple_loop()
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @simple_caller() {
  llvm.call @simple_loop() : () -> ()
  llvm.return
}

//func @simple_indirect_caller() {
//^bb0:
//  %f = constant @simple_loop : () -> ()
//  call_indirect %f() : () -> ()
//  return
//}

// CHECK-LABEL: define void @ml_caller()
// CHECK-NEXT:   call void @simple_loop()
// CHECK-NEXT:   call void @more_imperfectly_nested_loops()
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @ml_caller() {
  llvm.call @simple_loop() : () -> ()
  llvm.call @more_imperfectly_nested_loops() : () -> ()
  llvm.return
}

// CHECK-LABEL: declare i64 @body_args(i64)
llvm.func @body_args(i64) -> i64
// CHECK-LABEL: declare i32 @other(i64, i32)
llvm.func @other(i64, i32) -> i32

// CHECK-LABEL: define i32 @func_args(i32 {{%.*}}, i32 {{%.*}})
// CHECK-NEXT: br label %[[ARGS_bb1:[0-9]+]]
llvm.func @func_args(%arg0: i32, %arg1: i32) -> i32 {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.br ^bb1

// CHECK: [[ARGS_bb1]]:
// CHECK-NEXT: br label %[[ARGS_bb2:[0-9]+]]
^bb1:   // pred: ^bb0
  %1 = llvm.mlir.constant(0 : index) : i64
  %2 = llvm.mlir.constant(42 : index) : i64
  llvm.br ^bb2(%1 : i64)

// CHECK: [[ARGS_bb2]]:
// CHECK-NEXT:   %5 = phi i64 [ %12, %[[ARGS_bb3:[0-9]+]] ], [ 0, %[[ARGS_bb1]] ]
// CHECK-NEXT:   %6 = icmp slt i64 %5, 42
// CHECK-NEXT:   br i1 %6, label %[[ARGS_bb3]], label %[[ARGS_bb4:[0-9]+]]
^bb2(%3: i64): // 2 preds: ^bb1, ^bb3
  %4 = llvm.icmp "slt" %3, %2 : i64
  llvm.cond_br %4, ^bb3, ^bb4

// CHECK: [[ARGS_bb3]]:
// CHECK-NEXT:   %8 = call i64 @body_args(i64 %5)
// CHECK-NEXT:   %9 = call i32 @other(i64 %8, i32 %0)
// CHECK-NEXT:   %10 = call i32 @other(i64 %8, i32 %9)
// CHECK-NEXT:   %11 = call i32 @other(i64 %8, i32 %1)
// CHECK-NEXT:   %12 = add i64 %5, 1
// CHECK-NEXT:   br label %[[ARGS_bb2]]
^bb3:   // pred: ^bb2
  %5 = llvm.call @body_args(%3) : (i64) -> i64
  %6 = llvm.call @other(%5, %arg0) : (i64, i32) -> i32
  %7 = llvm.call @other(%5, %6) : (i64, i32) -> i32
  %8 = llvm.call @other(%5, %arg1) : (i64, i32) -> i32
  %9 = llvm.mlir.constant(1 : index) : i64
  %10 = llvm.add %3, %9 : i64
  llvm.br ^bb2(%10 : i64)

// CHECK: [[ARGS_bb4]]:
// CHECK-NEXT:   %14 = call i32 @other(i64 0, i32 0)
// CHECK-NEXT:   ret i32 %14
^bb4:   // pred: ^bb2
  %11 = llvm.mlir.constant(0 : index) : i64
  %12 = llvm.call @other(%11, %0) : (i64, i32) -> i32
  llvm.return %12 : i32
}

// CHECK: declare void @pre(i64)
llvm.func @pre(i64)

// CHECK: declare void @body2(i64, i64)
llvm.func @body2(i64, i64)

// CHECK: declare void @post(i64)
llvm.func @post(i64)

// CHECK-LABEL: define void @imperfectly_nested_loops()
// CHECK-NEXT:   br label %[[IMPER_bb1:[0-9]+]]
llvm.func @imperfectly_nested_loops() {
  llvm.br ^bb1

// CHECK: [[IMPER_bb1]]:
// CHECK-NEXT:   br label %[[IMPER_bb2:[0-9]+]]
^bb1:   // pred: ^bb0
  %0 = llvm.mlir.constant(0 : index) : i64
  %1 = llvm.mlir.constant(42 : index) : i64
  llvm.br ^bb2(%0 : i64)

// CHECK: [[IMPER_bb2]]:
// CHECK-NEXT:   %3 = phi i64 [ %13, %[[IMPER_bb7:[0-9]+]] ], [ 0, %[[IMPER_bb1]] ]
// CHECK-NEXT:   %4 = icmp slt i64 %3, 42
// CHECK-NEXT:   br i1 %4, label %[[IMPER_bb3:[0-9]+]], label %[[IMPER_bb8:[0-9]+]]
^bb2(%2: i64): // 2 preds: ^bb1, ^bb7
  %3 = llvm.icmp "slt" %2, %1 : i64
  llvm.cond_br %3, ^bb3, ^bb8

// CHECK: [[IMPER_bb3]]:
// CHECK-NEXT:   call void @pre(i64 %3)
// CHECK-NEXT:   br label %[[IMPER_bb4:[0-9]+]]
^bb3:   // pred: ^bb2
  llvm.call @pre(%2) : (i64) -> ()
  llvm.br ^bb4

// CHECK: [[IMPER_bb4]]:
// CHECK-NEXT:   br label %[[IMPER_bb5:[0-9]+]]
^bb4:   // pred: ^bb3
  %4 = llvm.mlir.constant(7 : index) : i64
  %5 = llvm.mlir.constant(56 : index) : i64
  llvm.br ^bb5(%4 : i64)

// CHECK: [[IMPER_bb5]]:
// CHECK-NEXT:   %8 = phi i64 [ %11, %[[IMPER_bb6:[0-9]+]] ], [ 7, %[[IMPER_bb4]] ]
// CHECK-NEXT:   %9 = icmp slt i64 %8, 56
// CHECK-NEXT:   br i1 %9, label %[[IMPER_bb6]], label %[[IMPER_bb7]]
^bb5(%6: i64): // 2 preds: ^bb4, ^bb6
  %7 = llvm.icmp "slt" %6, %5 : i64
  llvm.cond_br %7, ^bb6, ^bb7

// CHECK: [[IMPER_bb6]]:
// CHECK-NEXT:   call void @body2(i64 %3, i64 %8)
// CHECK-NEXT:   %11 = add i64 %8, 2
// CHECK-NEXT:   br label %[[IMPER_bb5]]
^bb6:   // pred: ^bb5
  llvm.call @body2(%2, %6) : (i64, i64) -> ()
  %8 = llvm.mlir.constant(2 : index) : i64
  %9 = llvm.add %6, %8 : i64
  llvm.br ^bb5(%9 : i64)

// CHECK: [[IMPER_bb7]]:
// CHECK-NEXT:   call void @post(i64 %3)
// CHECK-NEXT:   %13 = add i64 %3, 1
// CHECK-NEXT:   br label %[[IMPER_bb2]]
^bb7:   // pred: ^bb5
  llvm.call @post(%2) : (i64) -> ()
  %10 = llvm.mlir.constant(1 : index) : i64
  %11 = llvm.add %2, %10 : i64
  llvm.br ^bb2(%11 : i64)

// CHECK: [[IMPER_bb8]]:
// CHECK-NEXT:   ret void
^bb8:   // pred: ^bb2
  llvm.return
}

// CHECK: declare void @mid(i64)
llvm.func @mid(i64)

// CHECK: declare void @body3(i64, i64)
llvm.func @body3(i64, i64)

// A complete function transformation check.
// CHECK-LABEL: define void @more_imperfectly_nested_loops()
// CHECK-NEXT:   br label %1
// CHECK: 1:                                      ; preds = %0
// CHECK-NEXT:   br label %2
// CHECK: 2:                                      ; preds = %19, %1
// CHECK-NEXT:   %3 = phi i64 [ %20, %19 ], [ 0, %1 ]
// CHECK-NEXT:   %4 = icmp slt i64 %3, 42
// CHECK-NEXT:   br i1 %4, label %5, label %21
// CHECK: 5:                                      ; preds = %2
// CHECK-NEXT:   call void @pre(i64 %3)
// CHECK-NEXT:   br label %6
// CHECK: 6:                                      ; preds = %5
// CHECK-NEXT:   br label %7
// CHECK: 7:                                      ; preds = %10, %6
// CHECK-NEXT:   %8 = phi i64 [ %11, %10 ], [ 7, %6 ]
// CHECK-NEXT:   %9 = icmp slt i64 %8, 56
// CHECK-NEXT:   br i1 %9, label %10, label %12
// CHECK: 10:                                     ; preds = %7
// CHECK-NEXT:   call void @body2(i64 %3, i64 %8)
// CHECK-NEXT:   %11 = add i64 %8, 2
// CHECK-NEXT:   br label %7
// CHECK: 12:                                     ; preds = %7
// CHECK-NEXT:   call void @mid(i64 %3)
// CHECK-NEXT:   br label %13
// CHECK: 13:                                     ; preds = %12
// CHECK-NEXT:   br label %14
// CHECK: 14:                                     ; preds = %17, %13
// CHECK-NEXT:   %15 = phi i64 [ %18, %17 ], [ 18, %13 ]
// CHECK-NEXT:   %16 = icmp slt i64 %15, 37
// CHECK-NEXT:   br i1 %16, label %17, label %19
// CHECK: 17:                                     ; preds = %14
// CHECK-NEXT:   call void @body3(i64 %3, i64 %15)
// CHECK-NEXT:   %18 = add i64 %15, 3
// CHECK-NEXT:   br label %14
// CHECK: 19:                                     ; preds = %14
// CHECK-NEXT:   call void @post(i64 %3)
// CHECK-NEXT:   %20 = add i64 %3, 1
// CHECK-NEXT:   br label %2
// CHECK: 21:                                     ; preds = %2
// CHECK-NEXT:   ret void
// CHECK-NEXT: }
llvm.func @more_imperfectly_nested_loops() {
  llvm.br ^bb1
^bb1:	// pred: ^bb0
  %0 = llvm.mlir.constant(0 : index) : i64
  %1 = llvm.mlir.constant(42 : index) : i64
  llvm.br ^bb2(%0 : i64)
^bb2(%2: i64):	// 2 preds: ^bb1, ^bb11
  %3 = llvm.icmp "slt" %2, %1 : i64
  llvm.cond_br %3, ^bb3, ^bb12
^bb3:	// pred: ^bb2
  llvm.call @pre(%2) : (i64) -> ()
  llvm.br ^bb4
^bb4:	// pred: ^bb3
  %4 = llvm.mlir.constant(7 : index) : i64
  %5 = llvm.mlir.constant(56 : index) : i64
  llvm.br ^bb5(%4 : i64)
^bb5(%6: i64):	// 2 preds: ^bb4, ^bb6
  %7 = llvm.icmp "slt" %6, %5 : i64
  llvm.cond_br %7, ^bb6, ^bb7
^bb6:	// pred: ^bb5
  llvm.call @body2(%2, %6) : (i64, i64) -> ()
  %8 = llvm.mlir.constant(2 : index) : i64
  %9 = llvm.add %6, %8 : i64
  llvm.br ^bb5(%9 : i64)
^bb7:	// pred: ^bb5
  llvm.call @mid(%2) : (i64) -> ()
  llvm.br ^bb8
^bb8:	// pred: ^bb7
  %10 = llvm.mlir.constant(18 : index) : i64
  %11 = llvm.mlir.constant(37 : index) : i64
  llvm.br ^bb9(%10 : i64)
^bb9(%12: i64):	// 2 preds: ^bb8, ^bb10
  %13 = llvm.icmp "slt" %12, %11 : i64
  llvm.cond_br %13, ^bb10, ^bb11
^bb10:	// pred: ^bb9
  llvm.call @body3(%2, %12) : (i64, i64) -> ()
  %14 = llvm.mlir.constant(3 : index) : i64
  %15 = llvm.add %12, %14 : i64
  llvm.br ^bb9(%15 : i64)
^bb11:	// pred: ^bb9
  llvm.call @post(%2) : (i64) -> ()
  %16 = llvm.mlir.constant(1 : index) : i64
  %17 = llvm.add %2, %16 : i64
  llvm.br ^bb2(%17 : i64)
^bb12:	// pred: ^bb2
  llvm.return
}


//
// Check that linkage is translated for functions. No need to check all linkage
// flags since the logic is the same as for globals.
//

// CHECK: define internal void @func_internal
llvm.func internal @func_internal() {
  llvm.return
}

//
// Visibility attribute.
//

// CHECK-LABEL: define hidden void @hidden_func()
llvm.func hidden @hidden_func() {
  llvm.return
}

// CHECK-LABEL: define protected void @protected_func()
llvm.func protected @protected_func() {
  llvm.return
}

//
// dso_local attribute.
//

// CHECK: define dso_local void @dso_local_func
llvm.func @dso_local_func() attributes {dso_local} {
  llvm.return
}

//
// MemRef type conversion, allocation and communication with functions.
//

// CHECK-LABEL: define void @memref_alloc()
llvm.func @memref_alloc() {
// CHECK-NEXT: %{{[0-9]+}} = call ptr @malloc(i64 400)
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { ptr } undef, ptr %{{[0-9]+}}, 0
  %0 = llvm.mlir.constant(10 : index) : i64
  %1 = llvm.mlir.constant(10 : index) : i64
  %2 = llvm.mul %0, %1 : i64
  %3 = llvm.mlir.undef : !llvm.struct<(ptr)>
  %4 = llvm.mlir.constant(4 : index) : i64
  %5 = llvm.mul %2, %4 : i64
  %6 = llvm.call @malloc(%5) : (i64) -> !llvm.ptr
  %7 = llvm.insertvalue %6, %3[0] : !llvm.struct<(ptr)>
// CHECK-NEXT: ret void
  llvm.return
}

// CHECK-LABEL: declare i64 @get_index()
llvm.func @get_index() -> i64

// CHECK-LABEL: define void @store_load_static()
llvm.func @store_load_static() {
^bb0:
// CHECK-NEXT: %{{[0-9]+}} = call ptr @malloc(i64 40)
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { ptr } undef, ptr %{{[0-9]+}}, 0
  %0 = llvm.mlir.constant(10 : index) : i64
  %1 = llvm.mlir.undef : !llvm.struct<(ptr)>
  %2 = llvm.mlir.constant(4 : index) : i64
  %3 = llvm.mul %0, %2 : i64
  %4 = llvm.call @malloc(%3) : (i64) -> !llvm.ptr
  %6 = llvm.insertvalue %4, %1[0] : !llvm.struct<(ptr)>
  %7 = llvm.mlir.constant(1.000000e+00 : f32) : f32
  llvm.br ^bb1
^bb1:   // pred: ^bb0
  %8 = llvm.mlir.constant(0 : index) : i64
  %9 = llvm.mlir.constant(10 : index) : i64
  llvm.br ^bb2(%8 : i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb2(%10: i64):        // 2 preds: ^bb1, ^bb3
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 10
  %11 = llvm.icmp "slt" %10, %9 : i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %11, ^bb3, ^bb4
^bb3:   // pred: ^bb2
// CHECK: %{{[0-9]+}} = extractvalue { ptr } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, ptr %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 1.000000e+00, ptr %{{[0-9]+}}
  %12 = llvm.mlir.constant(10 : index) : i64
  %13 = llvm.extractvalue %6[0] : !llvm.struct<(ptr)>
  %14 = llvm.getelementptr %13[%10] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %7, %14 : f32, !llvm.ptr
  %15 = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %16 = llvm.add %10, %15 : i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb2(%16 : i64)
^bb4:   // pred: ^bb2
  llvm.br ^bb5
^bb5:   // pred: ^bb4
  %17 = llvm.mlir.constant(0 : index) : i64
  %18 = llvm.mlir.constant(10 : index) : i64
  llvm.br ^bb6(%17 : i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb6(%19: i64):        // 2 preds: ^bb5, ^bb7
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 10
  %20 = llvm.icmp "slt" %19, %18 : i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %20, ^bb7, ^bb8
^bb7:   // pred: ^bb6
// CHECK:      %{{[0-9]+}} = extractvalue { ptr } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, ptr %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, ptr %{{[0-9]+}}
  %21 = llvm.mlir.constant(10 : index) : i64
  %22 = llvm.extractvalue %6[0] : !llvm.struct<(ptr)>
  %23 = llvm.getelementptr %22[%19] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %24 = llvm.load %23 : !llvm.ptr -> f32
  %25 = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %26 = llvm.add %19, %25 : i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb6(%26 : i64)
^bb8:   // pred: ^bb6
// CHECK: ret void
  llvm.return
}

// CHECK-LABEL: define void @store_load_dynamic(i64 {{%.*}})
llvm.func @store_load_dynamic(%arg0: i64) {
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call ptr @malloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { ptr, i64 } undef, ptr %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { ptr, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
  %0 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %1 = llvm.mlir.constant(4 : index) : i64
  %2 = llvm.mul %arg0, %1 : i64
  %3 = llvm.call @malloc(%2) : (i64) -> !llvm.ptr
  %5 = llvm.insertvalue %3, %0[0] : !llvm.struct<(ptr, i64)>
  %6 = llvm.insertvalue %arg0, %5[1] : !llvm.struct<(ptr, i64)>
  %7 = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb1
^bb1:   // pred: ^bb0
  %8 = llvm.mlir.constant(0 : index) : i64
  llvm.br ^bb2(%8 : i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb2(%9: i64): // 2 preds: ^bb1, ^bb3
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, %{{[0-9]+}}
  %10 = llvm.icmp "slt" %9, %arg0 : i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %10, ^bb3, ^bb4
^bb3:   // pred: ^bb2
// CHECK:      %{{[0-9]+}} = extractvalue { ptr, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, ptr %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 1.000000e+00, ptr %{{[0-9]+}}
  %11 = llvm.extractvalue %6[1] : !llvm.struct<(ptr, i64)>
  %12 = llvm.extractvalue %6[0] : !llvm.struct<(ptr, i64)>
  %13 = llvm.getelementptr %12[%9] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %7, %13 : f32, !llvm.ptr
  %14 = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %15 = llvm.add %9, %14 : i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb2(%15 : i64)
^bb4:   // pred: ^bb3
  llvm.br ^bb5
^bb5:   // pred: ^bb4
  %16 = llvm.mlir.constant(0 : index) : i64
  llvm.br ^bb6(%16 : i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb6(%17: i64):        // 2 preds: ^bb5, ^bb7
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, %{{[0-9]+}}
  %18 = llvm.icmp "slt" %17, %arg0 : i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %18, ^bb7, ^bb8
^bb7:   // pred: ^bb6
// CHECK:      %{{[0-9]+}} = extractvalue { ptr, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, ptr %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, ptr %{{[0-9]+}}
  %19 = llvm.extractvalue %6[1] : !llvm.struct<(ptr, i64)>
  %20 = llvm.extractvalue %6[0] : !llvm.struct<(ptr, i64)>
  %21 = llvm.getelementptr %20[%17] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %22 = llvm.load %21 : !llvm.ptr -> f32
  %23 = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %24 = llvm.add %17, %23 : i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb6(%24 : i64)
^bb8:   // pred: ^bb6
// CHECK: ret void
  llvm.return
}

// CHECK-LABEL: define void @store_load_mixed(i64 {{%.*}})
llvm.func @store_load_mixed(%arg0: i64) {
  %0 = llvm.mlir.constant(10 : index) : i64
// CHECK-NEXT: %{{[0-9]+}} = mul i64 2, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 10
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call ptr @malloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { ptr, i64, i64 } undef, ptr %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { ptr, i64, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { ptr, i64, i64 } %{{[0-9]+}}, i64 10, 2
  %1 = llvm.mlir.constant(2 : index) : i64
  %2 = llvm.mlir.constant(4 : index) : i64
  %3 = llvm.mul %1, %arg0 : i64
  %4 = llvm.mul %3, %2 : i64
  %5 = llvm.mul %4, %0 : i64
  %6 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i64)>
  %7 = llvm.mlir.constant(4 : index) : i64
  %8 = llvm.mul %5, %7 : i64
  %9 = llvm.call @malloc(%8) : (i64) -> !llvm.ptr
  %11 = llvm.insertvalue %9, %6[0] : !llvm.struct<(ptr, i64, i64)>
  %12 = llvm.insertvalue %arg0, %11[1] : !llvm.struct<(ptr, i64, i64)>
  %13 = llvm.insertvalue %0, %12[2] : !llvm.struct<(ptr, i64, i64)>

// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
  %14 = llvm.mlir.constant(1 : index) : i64
  %15 = llvm.mlir.constant(2 : index) : i64
  %16 = llvm.call @get_index() : () -> i64
  %17 = llvm.call @get_index() : () -> i64
  %18 = llvm.mlir.constant(4.200000e+01 : f32) : f32
  %19 = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64, i64 } %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 1, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, ptr %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 4.200000e+01, ptr %{{[0-9]+}}
  %20 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, i64, i64)>
  %21 = llvm.mlir.constant(4 : index) : i64
  %22 = llvm.extractvalue %13[2] : !llvm.struct<(ptr, i64, i64)>
  %23 = llvm.mul %14, %20 : i64
  %24 = llvm.add %23, %15 : i64
  %25 = llvm.mul %24, %21 : i64
  %26 = llvm.add %25, %16 : i64
  %27 = llvm.mul %26, %22 : i64
  %28 = llvm.add %27, %17 : i64
  %29 = llvm.extractvalue %13[0] : !llvm.struct<(ptr, i64, i64)>
  %30 = llvm.getelementptr %29[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %18, %30 : f32, !llvm.ptr
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64, i64 } %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, ptr %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, ptr %{{[0-9]+}}
  %31 = llvm.mlir.constant(2 : index) : i64
  %32 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, i64, i64)>
  %33 = llvm.mlir.constant(4 : index) : i64
  %34 = llvm.extractvalue %13[2] : !llvm.struct<(ptr, i64, i64)>
  %35 = llvm.mul %17, %32 : i64
  %36 = llvm.add %35, %16 : i64
  %37 = llvm.mul %36, %33 : i64
  %38 = llvm.add %37, %15 : i64
  %39 = llvm.mul %38, %34 : i64
  %40 = llvm.add %39, %14 : i64
  %41 = llvm.extractvalue %13[0] : !llvm.struct<(ptr, i64, i64)>
  %42 = llvm.getelementptr %41[%40] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %43 = llvm.load %42 : !llvm.ptr -> f32
// CHECK-NEXT: ret void
  llvm.return
}

// CHECK-LABEL: define { ptr, i64 } @memref_args_rets({ ptr } {{%.*}}, { ptr, i64 } {{%.*}}, { ptr, i64 } {{%.*}})
llvm.func @memref_args_rets(%arg0: !llvm.struct<(ptr)>, %arg1: !llvm.struct<(ptr, i64)>, %arg2: !llvm.struct<(ptr, i64)>) -> !llvm.struct<(ptr, i64)> {
  %0 = llvm.mlir.constant(7 : index) : i64
// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
  %1 = llvm.call @get_index() : () -> i64
  %2 = llvm.mlir.constant(4.200000e+01 : f32) : f32
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, ptr %{{[0-9]+}}, i64 7
// CHECK-NEXT: store float 4.200000e+01, ptr %{{[0-9]+}}
  %3 = llvm.mlir.constant(10 : index) : i64
  %4 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr)>
  %5 = llvm.getelementptr %4[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %2, %5 : f32, !llvm.ptr
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, ptr %{{[0-9]+}}, i64 7
// CHECK-NEXT: store float 4.200000e+01, ptr %{{[0-9]+}}
  %6 = llvm.extractvalue %arg1[1] : !llvm.struct<(ptr, i64)>
  %7 = llvm.extractvalue %arg1[0] : !llvm.struct<(ptr, i64)>
  %8 = llvm.getelementptr %7[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %2, %8 : f32, !llvm.ptr
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = mul i64 7, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { ptr, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, ptr %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 4.200000e+01, ptr %{{[0-9]+}}
  %9 = llvm.mlir.constant(10 : index) : i64
  %10 = llvm.extractvalue %arg2[1] : !llvm.struct<(ptr, i64)>
  %11 = llvm.mul %0, %10 : i64
  %12 = llvm.add %11, %1 : i64
  %13 = llvm.extractvalue %arg2[0] : !llvm.struct<(ptr, i64)>
  %14 = llvm.getelementptr %13[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  llvm.store %2, %14 : f32, !llvm.ptr
// CHECK-NEXT: %{{[0-9]+}} = mul i64 10, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call ptr @malloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { ptr, i64 } undef, ptr %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { ptr, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
  %15 = llvm.mlir.constant(10 : index) : i64
  %16 = llvm.mul %15, %1 : i64
  %17 = llvm.mlir.undef : !llvm.struct<(ptr, i64)>
  %18 = llvm.mlir.constant(4 : index) : i64
  %19 = llvm.mul %16, %18 : i64
  %20 = llvm.call @malloc(%19) : (i64) -> !llvm.ptr
  %22 = llvm.insertvalue %20, %17[0] : !llvm.struct<(ptr, i64)>
  %23 = llvm.insertvalue %1, %22[1] : !llvm.struct<(ptr, i64)>
// CHECK-NEXT: ret { ptr, i64 } %{{[0-9]+}}
  llvm.return %23 : !llvm.struct<(ptr, i64)>
}


// CHECK-LABEL: define i64 @memref_dim({ ptr, i64, i64 } {{%.*}})
llvm.func @memref_dim(%arg0: !llvm.struct<(ptr, i64, i64)>) -> i64 {
// Expecting this to create an LLVM constant.
  %0 = llvm.mlir.constant(42 : index) : i64
// CHECK-NEXT: %2 = extractvalue { ptr, i64, i64 } %0, 1
  %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr, i64, i64)>
// Expecting this to create an LLVM constant.
  %2 = llvm.mlir.constant(10 : index) : i64
// CHECK-NEXT: %3 = extractvalue { ptr, i64, i64 } %0, 2
  %3 = llvm.extractvalue %arg0[2] : !llvm.struct<(ptr, i64, i64)>
// Checking that the constant for d0 has been created.
// CHECK-NEXT: %4 = add i64 42, %2
  %4 = llvm.add %0, %1 : i64
// Checking that the constant for d2 has been created.
// CHECK-NEXT: %5 = add i64 10, %3
  %5 = llvm.add %2, %3 : i64
// CHECK-NEXT: %6 = add i64 %4, %5
  %6 = llvm.add %4, %5 : i64
// CHECK-NEXT: ret i64 %6
  llvm.return %6 : i64
}

llvm.func @get_i64() -> i64
llvm.func @get_f32() -> f32
llvm.func @get_memref() -> !llvm.struct<(ptr, i64, i64)>

// CHECK-LABEL: define { i64, float, { ptr, i64, i64 } } @multireturn()
llvm.func @multireturn() -> !llvm.struct<(i64, f32, struct<(ptr, i64, i64)>)> {
  %0 = llvm.call @get_i64() : () -> i64
  %1 = llvm.call @get_f32() : () -> f32
  %2 = llvm.call @get_memref() : () -> !llvm.struct<(ptr, i64, i64)>
// CHECK:        %{{[0-9]+}} = insertvalue { i64, float, { ptr, i64, i64 } } undef, i64 %{{[0-9]+}}, 0
// CHECK-NEXT:   %{{[0-9]+}} = insertvalue { i64, float, { ptr, i64, i64 } } %{{[0-9]+}}, float %{{[0-9]+}}, 1
// CHECK-NEXT:   %{{[0-9]+}} = insertvalue { i64, float, { ptr, i64, i64 } } %{{[0-9]+}}, { ptr, i64, i64 } %{{[0-9]+}}, 2
// CHECK-NEXT:   ret { i64, float, { ptr, i64, i64 } } %{{[0-9]+}}
  %3 = llvm.mlir.undef : !llvm.struct<(i64, f32, struct<(ptr, i64, i64)>)>
  %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(i64, f32, struct<(ptr, i64, i64)>)>
  %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(i64, f32, struct<(ptr, i64, i64)>)>
  %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(i64, f32, struct<(ptr, i64, i64)>)>
  llvm.return %6 : !llvm.struct<(i64, f32, struct<(ptr, i64, i64)>)>
}


// CHECK-LABEL: define void @multireturn_caller()
llvm.func @multireturn_caller() {
// CHECK-NEXT:   %1 = call { i64, float, { ptr, i64, i64 } } @multireturn()
// CHECK-NEXT:   [[ret0:%[0-9]+]] = extractvalue { i64, float, { ptr, i64, i64 } } %1, 0
// CHECK-NEXT:   [[ret1:%[0-9]+]] = extractvalue { i64, float, { ptr, i64, i64 } } %1, 1
// CHECK-NEXT:   [[ret2:%[0-9]+]] = extractvalue { i64, float, { ptr, i64, i64 } } %1, 2
  %0 = llvm.call @multireturn() : () -> !llvm.struct<(i64, f32, struct<(ptr, i64, i64)>)>
  %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, f32, struct<(ptr, i64, i64)>)>
  %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, f32, struct<(ptr, i64, i64)>)>
  %3 = llvm.extractvalue %0[2] : !llvm.struct<(i64, f32, struct<(ptr, i64, i64)>)>
  %4 = llvm.mlir.constant(42) : i64
// CHECK:   add i64 [[ret0]], 42
  %5 = llvm.add %1, %4 : i64
  %6 = llvm.mlir.constant(4.200000e+01 : f32) : f32
// CHECK:   fadd float [[ret1]], 4.200000e+01
  %7 = llvm.fadd %2, %6 : f32
  %8 = llvm.mlir.constant(0 : index) : i64
  %9 = llvm.mlir.constant(42 : index) : i64
// CHECK:   extractvalue { ptr, i64, i64 } [[ret2]], 0
  %10 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, i64, i64)>
  %11 = llvm.mlir.constant(10 : index) : i64
  %12 = llvm.extractvalue %3[2] : !llvm.struct<(ptr, i64, i64)>
  %13 = llvm.mul %8, %10 : i64
  %14 = llvm.add %13, %8 : i64
  %15 = llvm.mul %14, %11 : i64
  %16 = llvm.add %15, %8 : i64
  %17 = llvm.mul %16, %12 : i64
  %18 = llvm.add %17, %8 : i64
  %19 = llvm.extractvalue %3[0] : !llvm.struct<(ptr, i64, i64)>
  %20 = llvm.getelementptr %19[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %21 = llvm.load %20 : !llvm.ptr -> f32
  llvm.return
}

// CHECK-LABEL: define <4 x float> @vector_ops(<4 x float> {{%.*}}, <4 x i1> {{%.*}}, <4 x i64> {{%.*}})
llvm.func @vector_ops(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: vector<4xi64>) -> vector<4xf32> {
  %0 = llvm.mlir.constant(dense<4.200000e+01> : vector<4xf32>) : vector<4xf32>
// CHECK-NEXT: %4 = fadd <4 x float> %0, <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
  %1 = llvm.fadd %arg0, %0 : vector<4xf32>
// CHECK-NEXT: %5 = select <4 x i1> %1, <4 x float> %4, <4 x float> %0
  %2 = llvm.select %arg1, %1, %arg0 : vector<4xi1>, vector<4xf32>
// CHECK-NEXT: %6 = sdiv <4 x i64> %2, %2
  %3 = llvm.sdiv %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT: %7 = udiv <4 x i64> %2, %2
  %4 = llvm.udiv %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT: %8 = srem <4 x i64> %2, %2
  %5 = llvm.srem %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT: %9 = urem <4 x i64> %2, %2
  %6 = llvm.urem %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT: %10 = fdiv <4 x float> %0, <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
  %7 = llvm.fdiv %arg0, %0 : vector<4xf32>
// CHECK-NEXT: %11 = frem <4 x float> %0, <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
  %8 = llvm.frem %arg0, %0 : vector<4xf32>
// CHECK-NEXT: %12 = and <4 x i64> %2, %2
  %9 = llvm.and %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT: %13 = or <4 x i64> %2, %2
  %10 = llvm.or %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT: %14 = xor <4 x i64> %2, %2
  %11 = llvm.xor %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT: %15 = shl <4 x i64> %2, %2
  %12 = llvm.shl %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT: %16 = lshr <4 x i64> %2, %2
  %13 = llvm.lshr %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT: %17 = ashr <4 x i64> %2, %2
  %14 = llvm.ashr %arg2, %arg2 : vector<4xi64>
// CHECK-NEXT:    ret <4 x float> %4
  llvm.return %1 : vector<4xf32>
}

// CHECK-LABEL: @vector_splat_1d
llvm.func @vector_splat_1d() -> vector<4xf32> {
  // CHECK: ret <4 x float> zeroinitializer
  %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<4xf32>) : vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: @vector_splat_1d_scalable
llvm.func @vector_splat_1d_scalable() -> vector<[4]xf32> {
  // CHECK: ret <vscale x 4 x float> zeroinitializer
  %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<[4]xf32>) : vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// CHECK-LABEL: @vector_splat_2d
llvm.func @vector_splat_2d() -> !llvm.array<4 x vector<16 x f32>> {
  // CHECK: ret [4 x <16 x float>] zeroinitializer
  %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<4x16xf32>) : !llvm.array<4 x vector<16 x f32>>
  llvm.return %0 : !llvm.array<4 x vector<16 x f32>>
}

// CHECK-LABEL: @vector_splat_3d
llvm.func @vector_splat_3d() -> !llvm.array<4 x array<16 x vector<4 x f32>>> {
  // CHECK: ret [4 x [16 x <4 x float>]] zeroinitializer
  %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<4x16x4xf32>) : !llvm.array<4 x array<16 x vector<4 x f32>>>
  llvm.return %0 : !llvm.array<4 x array<16 x vector<4 x f32>>>
}

// CHECK-LABEL: @vector_splat_nonzero
llvm.func @vector_splat_nonzero() -> vector<4xf32> {
  // CHECK: ret <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %0 = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: @vector_splat_nonzero_scalable
llvm.func @vector_splat_nonzero_scalable() -> vector<[4]xf32> {
  // CHECK: ret <vscale x 4 x float> shufflevector (<vscale x 4 x float> insertelement (<vscale x 4 x float> poison, float 1.000000e+00, i64 0), <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer)
  %0 = llvm.mlir.constant(dense<1.000000e+00> : vector<[4]xf32>) : vector<[4]xf32>
  llvm.return %0 : vector<[4]xf32>
}

// CHECK-LABEL: @ops
llvm.func @ops(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32) -> !llvm.struct<(f32, i32)> {
// CHECK-NEXT: fsub float %0, %1
  %0 = llvm.fsub %arg0, %arg1 : f32
// CHECK-NEXT: %6 = sub i32 %2, %3
  %1 = llvm.sub %arg2, %arg3 : i32
// CHECK-NEXT: %7 = icmp slt i32 %2, %6
  %2 = llvm.icmp "slt" %arg2, %1 : i32
// CHECK-NEXT: %8 = select i1 %7, i32 %2, i32 %6
  %3 = llvm.select %2, %arg2, %1 : i1, i32
// CHECK-NEXT: %9 = sdiv i32 %2, %3
  %4 = llvm.sdiv %arg2, %arg3 : i32
// CHECK-NEXT: %10 = udiv i32 %2, %3
  %5 = llvm.udiv %arg2, %arg3 : i32
// CHECK-NEXT: %11 = srem i32 %2, %3
  %6 = llvm.srem %arg2, %arg3 : i32
// CHECK-NEXT: %12 = urem i32 %2, %3
  %7 = llvm.urem %arg2, %arg3 : i32

  %8 = llvm.mlir.undef : !llvm.struct<(f32, i32)>
  %9 = llvm.insertvalue %0, %8[0] : !llvm.struct<(f32, i32)>
  %10 = llvm.insertvalue %3, %9[1] : !llvm.struct<(f32, i32)>

// CHECK: %15 = fdiv float %0, %1
  %11 = llvm.fdiv %arg0, %arg1 : f32
// CHECK-NEXT: %16 = frem float %0, %1
  %12 = llvm.frem %arg0, %arg1 : f32

// CHECK-NEXT: %17 = and i32 %2, %3
  %13 = llvm.and %arg2, %arg3 : i32
// CHECK-NEXT: %18 = or i32 %2, %3
  %14 = llvm.or %arg2, %arg3 : i32
// CHECK-NEXT: %19 = xor i32 %2, %3
  %15 = llvm.xor %arg2, %arg3 : i32
// CHECK-NEXT: %20 = shl i32 %2, %3
  %16 = llvm.shl %arg2, %arg3 : i32
// CHECK-NEXT: %21 = lshr i32 %2, %3
  %17 = llvm.lshr %arg2, %arg3 : i32
// CHECK-NEXT: %22 = ashr i32 %2, %3
  %18 = llvm.ashr %arg2, %arg3 : i32

// CHECK-NEXT: fneg float %0
  %19 = llvm.fneg %arg0 : f32

  llvm.return %10 : !llvm.struct<(f32, i32)>
}

// CHECK-LABEL: @gep
llvm.func @gep(%ptr: !llvm.ptr, %idx: i64,
               %ptr2: !llvm.ptr) {
  // CHECK: = getelementptr { i32, { i32, float } }, ptr %{{.*}}, i64 %{{.*}}, i32 1, i32 0
  llvm.getelementptr %ptr[%idx, 1, 0] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, struct<(i32, f32)>)>
  // CHECK: = getelementptr inbounds { [10 x float] }, ptr %{{.*}}, i64 %{{.*}}, i32 0, i64 %{{.*}}
  llvm.getelementptr inbounds %ptr2[%idx, 0, %idx] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.struct<(array<10 x f32>)>
  llvm.return
}

//
// Indirect function calls
//

// CHECK-LABEL: define void @indirect_const_call(i64 {{%.*}})
llvm.func @indirect_const_call(%arg0: i64) {
// CHECK-NEXT:  call void @body(i64 %0)
  %0 = llvm.mlir.addressof @body : !llvm.ptr
  llvm.call %0(%arg0) : !llvm.ptr, (i64) -> ()
// CHECK-NEXT:  ret void
  llvm.return
}

// CHECK-LABEL: define i32 @indirect_call(ptr addrspace(42) {{%.*}}, float {{%.*}})
llvm.func @indirect_call(%arg0: !llvm.ptr<42>, %arg1: f32) -> i32 {
// CHECK-NEXT:  %3 = call addrspace(42) i32 %0(float %1)
  %0 = llvm.call %arg0(%arg1) : !llvm.ptr<42>, (f32) -> i32
// CHECK-NEXT:  ret i32 %3
  llvm.return %0 : i32
}

//
// Check that we properly construct phi nodes in the blocks that have the same
// predecessor more than once.
//

// CHECK-LABEL: define void @cond_br_arguments(i1 {{%.*}}, i1 {{%.*}})
llvm.func @cond_br_arguments(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   br i1 %0, label %3, label %5
  llvm.cond_br %arg0, ^bb1(%arg0 : i1), ^bb2

// CHECK:      3:
// CHECK-NEXT:   %4 = phi i1 [ %1, %5 ], [ %0, %2 ]
^bb1(%0 : i1):
// CHECK-NEXT:   ret void
  llvm.return

// CHECK:      5:
^bb2:
// CHECK-NEXT:   br label %3
  llvm.br ^bb1(%arg1 : i1)
}

// CHECK-LABEL: define void @llvm_noalias(ptr noalias {{%*.}})
llvm.func @llvm_noalias(%arg0: !llvm.ptr {llvm.noalias}) {
  llvm.return
}

// CHECK-LABEL: declare void @llvm_noalias_decl(ptr noalias)
llvm.func @llvm_noalias_decl(!llvm.ptr {llvm.noalias})

// CHECK-LABEL: define void @byrefattr(ptr byref(i32) %
llvm.func @byrefattr(%arg0: !llvm.ptr {llvm.byref = i32}) {
  llvm.return
}

// CHECK-LABEL: declare void @byrefattr_decl(ptr byref(i32))
llvm.func @byrefattr_decl(!llvm.ptr {llvm.byref = i32})

// CHECK-LABEL: define void @byvalattr(ptr byval(i32) %
llvm.func @byvalattr(%arg0: !llvm.ptr {llvm.byval = i32}) {
  llvm.return
}

// CHECK-LABEL: declare void @byvalattr_decl(ptr byval(i32))
llvm.func @byvalattr_decl(!llvm.ptr {llvm.byval = i32})

// CHECK-LABEL: define void @sretattr(ptr sret(i32) %
llvm.func @sretattr(%arg0: !llvm.ptr {llvm.sret = i32}) {
  llvm.return
}

// CHECK-LABEL: declare void @sretattr_decl(ptr sret(i32))
llvm.func @sretattr_decl(!llvm.ptr {llvm.sret = i32})

// CHECK-LABEL: define void @nestattr(ptr nest %
llvm.func @nestattr(%arg0: !llvm.ptr {llvm.nest}) {
  llvm.return
}

// CHECK-LABEL: declare void @nestattr_decl(ptr nest)
llvm.func @nestattr_decl(!llvm.ptr {llvm.nest})

// CHECK-LABEL: define void @noundefattr(i32 noundef %
llvm.func @noundefattr(%arg0: i32 {llvm.noundef}) {
  llvm.return
}

// CHECK-LABEL: declare void @noundefattr_decl(i32 noundef)
llvm.func @noundefattr_decl(i32 {llvm.noundef})

// CHECK-LABEL: define void @llvm_align(ptr align 4 {{%*.}})
llvm.func @llvm_align(%arg0: !llvm.ptr {llvm.align = 4}) {
  llvm.return
}

// CHECK-LABEL: declare void @llvm_align_decl(ptr align 4)
llvm.func @llvm_align_decl(!llvm.ptr {llvm.align = 4})

// CHECK-LABEL: define void @inallocaattr(ptr inalloca(i32) %
llvm.func @inallocaattr(%arg0: !llvm.ptr {llvm.inalloca = i32}) {
  llvm.return
}

// CHECK-LABEL: declare void @inallocaattr_decl(ptr inalloca(i32))
llvm.func @inallocaattr_decl(!llvm.ptr {llvm.inalloca = i32})

// CHECK-LABEL: define void @signextattr(i1 signext %
llvm.func @signextattr(%arg0: i1 {llvm.signext}) {
  llvm.return
}

// CHECK-LABEL: declare void @signextattr_decl(i1 signext)
llvm.func @signextattr_decl(i1 {llvm.signext})

// CHECK-LABEL: define void @zeroextattr(i1 zeroext %
llvm.func @zeroextattr(%arg0: i1 {llvm.zeroext}) {
  llvm.return
}

// CHECK-LABEL: declare void @zeroextattr_decl(i1 zeroext)
llvm.func @zeroextattr_decl(i1 {llvm.zeroext})

// CHECK-LABEL: declare void @alignattr_decl(ptr align 64)
llvm.func @alignattr_decl(!llvm.ptr {llvm.align = 64 : i64})

// CHECK-LABEL: declare void @dereferenceableattr_decl(ptr dereferenceable(32))
llvm.func @dereferenceableattr_decl(!llvm.ptr {llvm.dereferenceable = 32 : i64})

// CHECK-LABEL: declare void @dereferenceableornullattr_decl(ptr dereferenceable_or_null(32))
llvm.func @dereferenceableornullattr_decl(!llvm.ptr {llvm.dereferenceable_or_null = 32 : i64})

// CHECK-LABEL: declare void @inregattr_decl(ptr inreg)
llvm.func @inregattr_decl(!llvm.ptr {llvm.inreg})

// CHECK-LABEL: declare void @nocaptureattr_decl(ptr nocapture)
llvm.func @nocaptureattr_decl(!llvm.ptr {llvm.nocapture})

// CHECK-LABEL: declare void @nofreeattr_decl(ptr nofree)
llvm.func @nofreeattr_decl(!llvm.ptr {llvm.nofree})

// CHECK-LABEL: declare void @nonnullattr_decl(ptr nonnull)
llvm.func @nonnullattr_decl(!llvm.ptr {llvm.nonnull})

// CHECK-LABEL: declare void @preallocatedattr_decl(ptr preallocated(float))
llvm.func @preallocatedattr_decl(!llvm.ptr {llvm.preallocated = f32})

// CHECK-LABEL: declare ptr @returnedattr_decl(ptr returned)
llvm.func @returnedattr_decl(!llvm.ptr {llvm.returned}) -> !llvm.ptr

// CHECK-LABEL: declare void @alignstackattr_decl(ptr alignstack(32))
llvm.func @alignstackattr_decl(!llvm.ptr {llvm.alignstack = 32 : i64})

// CHECK-LABEL: declare void @writeonlyattr_decl(ptr writeonly)
llvm.func @writeonlyattr_decl(!llvm.ptr {llvm.writeonly})

// CHECK-LABEL: declare align 4 ptr @alignattr_ret_decl()
llvm.func @alignattr_ret_decl() -> (!llvm.ptr {llvm.align = 4})

// CHECK-LABEL: declare noalias ptr @noaliasattr_ret_decl()
llvm.func @noaliasattr_ret_decl() -> (!llvm.ptr {llvm.noalias})

// CHECK-LABEL: declare noundef ptr @noundefattr_ret_decl()
llvm.func @noundefattr_ret_decl() -> (!llvm.ptr {llvm.noundef})

// CHECK-LABEL: declare signext i1 @signextattr_ret_decl()
llvm.func @signextattr_ret_decl() -> (i1 {llvm.signext})

// CHECK-LABEL: declare zeroext i1 @zeroextattr_ret_decl()
llvm.func @zeroextattr_ret_decl() -> (i1 {llvm.zeroext})

// CHECK-LABEL: declare nonnull ptr @nonnullattr_ret_decl()
llvm.func @nonnullattr_ret_decl() -> (!llvm.ptr {llvm.nonnull})

// CHECK-LABEL: declare dereferenceable(32) ptr @dereferenceableattr_ret_decl()
llvm.func @dereferenceableattr_ret_decl() -> (!llvm.ptr {llvm.dereferenceable = 32 : i64})

// CHECK-LABEL: declare dereferenceable_or_null(16) ptr @dereferenceableornullattr_ret_decl()
llvm.func @dereferenceableornullattr_ret_decl() -> (!llvm.ptr {llvm.dereferenceable_or_null = 16 : i64})

// CHECK-LABEL: declare inreg ptr @inregattr_ret_decl()
llvm.func @inregattr_ret_decl() -> (!llvm.ptr {llvm.inreg})

// CHECK-LABEL: @varargs(...)
llvm.func @varargs(...) -> f32

// CHECK-LABEL: define void @varargs_call
llvm.func @varargs_call(%arg0 : i32) {
// CHECK:  call float (...) @varargs(i32 %{{.*}})
// CHECK:  call nnan float (...) @varargs(i32 %{{.*}})
  llvm.call @varargs(%arg0) vararg(!llvm.func<f32 (...)>) : (i32) -> (f32)
  llvm.call @varargs(%arg0) vararg(!llvm.func<f32 (...)>) {fastmathFlags = #llvm.fastmath<nnan>} : (i32) -> (f32)
  llvm.return
}

// CHECK-LABEL: define void @indirect_varargs_call(ptr %0, i32 %1)
llvm.func @indirect_varargs_call(%arg0 : !llvm.ptr, %arg1 : i32) {
// CHECK:  call float (...) %0(i32 %1)
// CHECK:  call nnan float (...) %0(i32 %1)
  llvm.call %arg0(%arg1) vararg(!llvm.func<f32 (...)>) : !llvm.ptr, (i32) -> (f32)
  llvm.call %arg0(%arg1) vararg(!llvm.func<f32 (...)>) {fastmathFlags = #llvm.fastmath<nnan>} : !llvm.ptr, (i32) -> (f32)
  llvm.return
}

llvm.func @intpointerconversion(%arg0 : i32) -> i32 {
// CHECK:      %2 = inttoptr i32 %0 to ptr
// CHECK-NEXT: %3 = ptrtoint ptr %2 to i32
  %1 = llvm.inttoptr %arg0 : i32 to !llvm.ptr
  %2 = llvm.ptrtoint %1 : !llvm.ptr to i32
  llvm.return %2 : i32
}

llvm.func @fpconversion(%arg0 : i32) -> i32 {
// CHECK:      %2 = sitofp i32 %0 to float
// CHECK-NEXT: %3 = fptosi float %2 to i32
// CHECK-NEXT: %4 = uitofp i32 %3 to float
// CHECK-NEXT: %5 = fptoui float %4 to i32
  %1 = llvm.sitofp %arg0 : i32 to f32
  %2 = llvm.fptosi %1 : f32 to i32
  %3 = llvm.uitofp %2 : i32 to f32
  %4 = llvm.fptoui %3 : f32 to i32
  llvm.return %4 : i32
}

// CHECK-LABEL: @addrspace
llvm.func @addrspace(%arg0 : !llvm.ptr) -> !llvm.ptr<2> {
// CHECK: %2 = addrspacecast ptr %0 to ptr addrspace(2)
  %1 = llvm.addrspacecast %arg0 : !llvm.ptr to !llvm.ptr<2>
  llvm.return %1 : !llvm.ptr<2>
}

llvm.func @stringconstant() -> !llvm.array<12 x i8> {
  %1 = llvm.mlir.constant("Hello world!") : !llvm.array<12 x i8>
  // CHECK: ret [12 x i8] c"Hello world!"
  llvm.return %1 : !llvm.array<12 x i8>
}

llvm.func @complexfpconstant() -> !llvm.struct<(f32, f32)> {
  %1 = llvm.mlir.constant([-1.000000e+00 : f32, 0.000000e+00 : f32]) : !llvm.struct<(f32, f32)>
  // CHECK: ret { float, float } { float -1.000000e+00, float 0.000000e+00 }
  llvm.return %1 : !llvm.struct<(f32, f32)>
}

llvm.func @complexintconstant() -> !llvm.struct<(i32, i32)> {
  %1 = llvm.mlir.constant([-1 : i32, 0 : i32]) : !llvm.struct<(i32, i32)>
  // CHECK: ret { i32, i32 } { i32 -1, i32 0 }
  llvm.return %1 : !llvm.struct<(i32, i32)>
}

llvm.func @complexintconstantsplat() -> !llvm.array<2 x !llvm.struct<(i32, i32)>> {
  %1 = llvm.mlir.constant(dense<(0, 1)> : tensor<complex<i32>>) : !llvm.array<2 x !llvm.struct<(i32, i32)>>
  // CHECK: ret [2 x { i32, i32 }] [{ i32, i32 } { i32 0, i32 1 }, { i32, i32 } { i32 0, i32 1 }]
  llvm.return %1 : !llvm.array<2 x !llvm.struct<(i32, i32)>>
}

llvm.func @complexintconstantarray() -> !llvm.array<2 x !llvm.array<2 x !llvm.struct<(i32, i32)>>> {
  %1 = llvm.mlir.constant(dense<[[(0, 1), (2, 3)], [(4, 5), (6, 7)]]> : tensor<2x2xcomplex<i32>>) : !llvm.array<2 x!llvm.array<2 x !llvm.struct<(i32, i32)>>>
  // CHECK{LITERAL}: ret [2 x [2 x { i32, i32 }]] [[2 x { i32, i32 }] [{ i32, i32 } { i32 0, i32 1 }, { i32, i32 } { i32 2, i32 3 }], [2 x { i32, i32 }] [{ i32, i32 } { i32 4, i32 5 }, { i32, i32 } { i32 6, i32 7 }]]
  llvm.return %1 : !llvm.array<2 x !llvm.array<2 x !llvm.struct<(i32, i32)>>>
}

// CHECK-LABEL: @indexconstantsplat
llvm.func @indexconstantsplat() -> vector<3xi32> {
  %1 = llvm.mlir.constant(dense<42> : vector<3xindex>) : vector<3xi32>
  // CHECK: ret <3 x i32> <i32 42, i32 42, i32 42>
  llvm.return %1 : vector<3xi32>
}

// CHECK-LABEL: @indexconstantarray
llvm.func @indexconstantarray() -> vector<3xi32> {
  %1 = llvm.mlir.constant(dense<[0, 1, 2]> : vector<3xindex>) : vector<3xi32>
  // CHECK: ret <3 x i32> <i32 0, i32 1, i32 2>
  llvm.return %1 : vector<3xi32>
}

llvm.func @noreach() {
// CHECK:    unreachable
  llvm.unreachable
}

// CHECK-LABEL: define void @fcmp
llvm.func @fcmp(%arg0: f32, %arg1: f32) {
  // CHECK: fcmp oeq float %0, %1
  // CHECK-NEXT: fcmp ogt float %0, %1
  // CHECK-NEXT: fcmp oge float %0, %1
  // CHECK-NEXT: fcmp olt float %0, %1
  // CHECK-NEXT: fcmp ole float %0, %1
  // CHECK-NEXT: fcmp one float %0, %1
  // CHECK-NEXT: fcmp ord float %0, %1
  // CHECK-NEXT: fcmp ueq float %0, %1
  // CHECK-NEXT: fcmp ugt float %0, %1
  // CHECK-NEXT: fcmp uge float %0, %1
  // CHECK-NEXT: fcmp ult float %0, %1
  // CHECK-NEXT: fcmp ule float %0, %1
  // CHECK-NEXT: fcmp une float %0, %1
  // CHECK-NEXT: fcmp uno float %0, %1
  %0 = llvm.fcmp "oeq" %arg0, %arg1 : f32
  %1 = llvm.fcmp "ogt" %arg0, %arg1 : f32
  %2 = llvm.fcmp "oge" %arg0, %arg1 : f32
  %3 = llvm.fcmp "olt" %arg0, %arg1 : f32
  %4 = llvm.fcmp "ole" %arg0, %arg1 : f32
  %5 = llvm.fcmp "one" %arg0, %arg1 : f32
  %6 = llvm.fcmp "ord" %arg0, %arg1 : f32
  %7 = llvm.fcmp "ueq" %arg0, %arg1 : f32
  %8 = llvm.fcmp "ugt" %arg0, %arg1 : f32
  %9 = llvm.fcmp "uge" %arg0, %arg1 : f32
  %10 = llvm.fcmp "ult" %arg0, %arg1 : f32
  %11 = llvm.fcmp "ule" %arg0, %arg1 : f32
  %12 = llvm.fcmp "une" %arg0, %arg1 : f32
  %13 = llvm.fcmp "uno" %arg0, %arg1 : f32
  llvm.return
}

// CHECK-LABEL: @vect
llvm.func @vect(%arg0: vector<4xf32>, %arg1: i32, %arg2: f32) {
  // CHECK-NEXT: extractelement <4 x float> {{.*}}, i32
  // CHECK-NEXT: insertelement <4 x float> {{.*}}, float %2, i32
  // CHECK-NEXT: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <5 x i32> <i32 0, i32 0, i32 0, i32 0, i32 7>
  %0 = llvm.extractelement %arg0[%arg1 : i32] : vector<4xf32>
  %1 = llvm.insertelement %arg2, %arg0[%arg1 : i32] : vector<4xf32>
  %2 = llvm.shufflevector %arg0, %arg0 [0, 0, 0, 0, 7] : vector<4xf32>
  llvm.return
}

// CHECK-LABEL: @vect_i64idx
llvm.func @vect_i64idx(%arg0: vector<4xf32>, %arg1: i64, %arg2: f32) {
  // CHECK-NEXT: extractelement <4 x float> {{.*}}, i64
  // CHECK-NEXT: insertelement <4 x float> {{.*}}, float %2, i64
  %0 = llvm.extractelement %arg0[%arg1 : i64] : vector<4xf32>
  %1 = llvm.insertelement %arg2, %arg0[%arg1 : i64] : vector<4xf32>
  llvm.return
}

// CHECK-LABEL: @scalable_vect
llvm.func @scalable_vect(%arg0: vector<[4]xf32>, %arg1: i32, %arg2: f32) {
  // CHECK-NEXT: extractelement <vscale x 4 x float> {{.*}}, i32
  // CHECK-NEXT: insertelement <vscale x 4 x float> {{.*}}, float %2, i32
  // CHECK-NEXT: shufflevector <vscale x 4 x float> %0, <vscale x 4 x float> %0, <vscale x 4 x i32> zeroinitializer
  %0 = llvm.extractelement %arg0[%arg1 : i32] : vector<[4]xf32>
  %1 = llvm.insertelement %arg2, %arg0[%arg1 : i32] : vector<[4]xf32>
  %2 = llvm.shufflevector %arg0, %arg0 [0, 0, 0, 0] : vector<[4]xf32>
  llvm.return
}

// CHECK-LABEL: @scalable_vect_i64idx
llvm.func @scalable_vect_i64idx(%arg0: vector<[4]xf32>, %arg1: i64, %arg2: f32) {
  // CHECK-NEXT: extractelement <vscale x 4 x float> {{.*}}, i64
  // CHECK-NEXT: insertelement <vscale x 4 x float> {{.*}}, float %2, i64
  %0 = llvm.extractelement %arg0[%arg1 : i64] : vector<[4]xf32>
  %1 = llvm.insertelement %arg2, %arg0[%arg1 : i64] : vector<[4]xf32>
  llvm.return
}

// CHECK-LABEL: @alloca
llvm.func @alloca(%size : i64) {
  // Alignment automatically set by the LLVM IR builder when alignment attribute
  // is 0.
  //  CHECK: alloca {{.*}} align 4
  llvm.alloca %size x i32 {alignment = 0} : (i64) -> (!llvm.ptr)
  // CHECK-NEXT: alloca {{.*}} align 8
  llvm.alloca %size x i32 {alignment = 8} : (i64) -> (!llvm.ptr)
  // CHECK-NEXT: alloca {{.*}} addrspace(3)
  llvm.alloca %size x i32 {alignment = 0} : (i64) -> (!llvm.ptr<3>)
  // CHECK-NEXT: alloca inalloca {{.*}} align 4
  llvm.alloca inalloca %size x i32 : (i64) -> !llvm.ptr
  llvm.return
}

// CHECK-LABEL: @constants
llvm.func @constants() -> vector<4xf32> {
  // CHECK: ret <4 x float> <float 4.2{{0*}}e+01, float 0.{{0*}}e+00, float 0.{{0*}}e+00, float 0.{{0*}}e+00>
  %0 = llvm.mlir.constant(sparse<[0], [4.2e+01]> : vector<4xf32>) : vector<4xf32>
  llvm.return %0 : vector<4xf32>
}

// CHECK-LABEL: @fp_casts
llvm.func @fp_casts(%fp1 : f32, %fp2 : f64) -> i16 {
// CHECK:    fptrunc double {{.*}} to float
  %a = llvm.fptrunc %fp2 : f64 to f32
// CHECK:    fpext float {{.*}} to double
  %b = llvm.fpext %fp1 : f32 to f64
// CHECK:    fptosi double {{.*}} to i16
  %c = llvm.fptosi %b : f64 to i16
  llvm.return %c : i16
}

// CHECK-LABEL: @integer_extension_and_truncation
llvm.func @integer_extension_and_truncation(%a : i32) {
// CHECK:    sext i32 {{.*}} to i64
// CHECK:    zext i32 {{.*}} to i64
// CHECK:    trunc i32 {{.*}} to i16
  %0 = llvm.sext %a : i32 to i64
  %1 = llvm.zext %a : i32 to i64
  %2 = llvm.trunc %a : i32 to i16
  llvm.return
}

// Check that the auxiliary `null` operation is converted into a `null` value.
// CHECK-LABEL: @null
llvm.func @null() -> !llvm.ptr {
  %0 = llvm.mlir.zero : !llvm.ptr
  // CHECK: ret ptr null
  llvm.return %0 : !llvm.ptr
}

// Check that dense elements attributes are exported properly in constants.
// CHECK-LABEL: @elements_constant_3d_vector
llvm.func @elements_constant_3d_vector() -> !llvm.array<2 x array<2 x vector<2 x i32>>> {
  // CHECK: ret [2 x [2 x <2 x i32>]]
  // CHECK-SAME: {{\[}}[2 x <2 x i32>] [<2 x i32> <i32 1, i32 2>, <2 x i32> <i32 3, i32 4>],
  // CHECK-SAME:       [2 x <2 x i32>] [<2 x i32> <i32 42, i32 43>, <2 x i32> <i32 44, i32 45>]]
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : vector<2x2x2xi32>) : !llvm.array<2 x array<2 x vector<2 x i32>>>
  llvm.return %0 : !llvm.array<2 x array<2 x vector<2 x i32>>>
}

// CHECK-LABEL: @elements_constant_3d_array
llvm.func @elements_constant_3d_array() -> !llvm.array<2 x array<2 x array<2 x i32>>> {
  // CHECK: ret [2 x [2 x [2 x i32]]]
  // CHECK-SAME: {{\[}}[2 x [2 x i32]] {{\[}}[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]],
  // CHECK-SAME:       [2 x [2 x i32]] {{\[}}[2 x i32] [i32 42, i32 43], [2 x i32] [i32 44, i32 45]]]
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : tensor<2x2x2xi32>) : !llvm.array<2 x array<2 x array<2 x i32>>>
  llvm.return %0 : !llvm.array<2 x array<2 x array<2 x i32>>>
}

// CHECK-LABEL: @atomicrmw
llvm.func @atomicrmw(
    %f32_ptr : !llvm.ptr, %f32 : f32,
    %i32_ptr : !llvm.ptr, %i32 : i32) {
  // CHECK: atomicrmw fadd ptr %{{.*}}, float %{{.*}} monotonic
  %0 = llvm.atomicrmw fadd %f32_ptr, %f32 monotonic : !llvm.ptr, f32
  // CHECK: atomicrmw fsub ptr %{{.*}}, float %{{.*}} monotonic
  %1 = llvm.atomicrmw fsub %f32_ptr, %f32 monotonic : !llvm.ptr, f32
  // CHECK: atomicrmw fmax ptr %{{.*}}, float %{{.*}} monotonic
  %2 = llvm.atomicrmw fmax %f32_ptr, %f32 monotonic : !llvm.ptr, f32
  // CHECK: atomicrmw fmin ptr %{{.*}}, float %{{.*}} monotonic
  %3 = llvm.atomicrmw fmin %f32_ptr, %f32 monotonic : !llvm.ptr, f32
  // CHECK: atomicrmw xchg ptr %{{.*}}, float %{{.*}} monotonic
  %4 = llvm.atomicrmw xchg %f32_ptr, %f32 monotonic : !llvm.ptr, f32
  // CHECK: atomicrmw add ptr %{{.*}}, i32 %{{.*}} acquire
  %5 = llvm.atomicrmw add %i32_ptr, %i32 acquire : !llvm.ptr, i32
  // CHECK: atomicrmw sub ptr %{{.*}}, i32 %{{.*}} release
  %6 = llvm.atomicrmw sub %i32_ptr, %i32 release : !llvm.ptr, i32
  // CHECK: atomicrmw and ptr %{{.*}}, i32 %{{.*}} acq_rel
  %7 = llvm.atomicrmw _and %i32_ptr, %i32 acq_rel : !llvm.ptr, i32
  // CHECK: atomicrmw nand ptr %{{.*}}, i32 %{{.*}} seq_cst
  %8 = llvm.atomicrmw nand %i32_ptr, %i32 seq_cst : !llvm.ptr, i32
  // CHECK: atomicrmw or ptr %{{.*}}, i32 %{{.*}} monotonic
  %9 = llvm.atomicrmw _or %i32_ptr, %i32 monotonic : !llvm.ptr, i32
  // CHECK: atomicrmw xor ptr %{{.*}}, i32 %{{.*}} monotonic
  %10 = llvm.atomicrmw _xor %i32_ptr, %i32 monotonic : !llvm.ptr, i32
  // CHECK: atomicrmw max ptr %{{.*}}, i32 %{{.*}} monotonic
  %11 = llvm.atomicrmw max %i32_ptr, %i32 monotonic : !llvm.ptr, i32
  // CHECK: atomicrmw min ptr %{{.*}}, i32 %{{.*}} monotonic
  %12 = llvm.atomicrmw min %i32_ptr, %i32 monotonic : !llvm.ptr, i32
  // CHECK: atomicrmw umax ptr %{{.*}}, i32 %{{.*}} monotonic
  %13 = llvm.atomicrmw umax %i32_ptr, %i32 monotonic : !llvm.ptr, i32
  // CHECK: atomicrmw umin ptr %{{.*}}, i32 %{{.*}} monotonic
  %14 = llvm.atomicrmw umin %i32_ptr, %i32 monotonic : !llvm.ptr, i32
  // CHECK: atomicrmw uinc_wrap ptr %{{.*}}, i32 %{{.*}} monotonic
  %15 = llvm.atomicrmw uinc_wrap %i32_ptr, %i32 monotonic : !llvm.ptr, i32
  // CHECK: atomicrmw udec_wrap ptr %{{.*}}, i32 %{{.*}} monotonic
  %16 = llvm.atomicrmw udec_wrap %i32_ptr, %i32 monotonic : !llvm.ptr, i32

  // CHECK: atomicrmw volatile
  // CHECK-SAME:  syncscope("singlethread")
  // CHECK-SAME:  align 8
  %17 = llvm.atomicrmw volatile udec_wrap %i32_ptr, %i32 syncscope("singlethread") monotonic {alignment = 8 : i64} : !llvm.ptr, i32
  llvm.return
}

// CHECK-LABEL: @cmpxchg
llvm.func @cmpxchg(%ptr : !llvm.ptr, %cmp : i32, %val: i32) {
  // CHECK: cmpxchg ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}} acq_rel monotonic
  %0 = llvm.cmpxchg %ptr, %cmp, %val acq_rel monotonic : !llvm.ptr, i32
  // CHECK: %{{[0-9]+}} = extractvalue { i32, i1 } %{{[0-9]+}}, 0
  %1 = llvm.extractvalue %0[0] : !llvm.struct<(i32, i1)>
  // CHECK: %{{[0-9]+}} = extractvalue { i32, i1 } %{{[0-9]+}}, 1
  %2 = llvm.extractvalue %0[1] : !llvm.struct<(i32, i1)>

  // CHECK:  cmpxchg weak volatile
  // CHECK-SAME:  syncscope("singlethread")
  // CHECK-SAME:  align 8
  %3 = llvm.cmpxchg weak volatile %ptr, %cmp, %val syncscope("singlethread") acq_rel monotonic {alignment = 8 : i64} : !llvm.ptr, i32
  llvm.return
}

llvm.mlir.global external constant @_ZTIi() : !llvm.ptr
llvm.func @foo(!llvm.ptr)
llvm.func @vararg_foo(!llvm.ptr, ...)
llvm.func @bar(!llvm.ptr) -> !llvm.ptr
llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: @invokeLandingpad
llvm.func @invokeLandingpad() -> i32 attributes { personality = @__gxx_personality_v0 } {
// CHECK: %[[a1:[0-9]+]] = alloca i8
  %0 = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(dense<0> : vector<1xi8>) : !llvm.array<1 x i8>
  %2 = llvm.mlir.addressof @_ZTIi : !llvm.ptr
  %4 = llvm.mlir.zero : !llvm.ptr
  %5 = llvm.mlir.constant(1 : i32) : i32
  %6 = llvm.alloca %5 x i8 : (i32) -> !llvm.ptr
// CHECK: invoke void @foo(ptr %[[a1]])
// CHECK-NEXT: to label %[[normal:[0-9]+]] unwind label %[[unwind:[0-9]+]]
  llvm.invoke @foo(%6) to ^bb2 unwind ^bb1 : (!llvm.ptr) -> ()

// CHECK: [[unwind]]:
^bb1:
// CHECK: %{{[0-9]+}} = landingpad { ptr, i32 }
// CHECK-NEXT:             catch ptr null
// CHECK-NEXT:             catch ptr @_ZTIi
// CHECK-NEXT:             filter [1 x i8] zeroinitializer
  %7 = llvm.landingpad (catch %4 : !llvm.ptr) (catch %2 : !llvm.ptr) (filter %1 : !llvm.array<1 x i8>) : !llvm.struct<(ptr, i32)>
// CHECK: br label %[[final:[0-9]+]]
  llvm.br ^bb3

// CHECK: [[normal]]:
// CHECK-NEXT: ret i32 1
^bb2:	// 2 preds: ^bb0, ^bb3
  llvm.return %5 : i32

// CHECK: [[final]]:
// CHECK-NEXT: %{{[0-9]+}} = invoke ptr @bar(ptr %[[a1]])
// CHECK-NEXT:          to label %[[normal]] unwind label %[[unwind]]
^bb3:	// pred: ^bb1
  %8 = llvm.invoke @bar(%6) to ^bb2 unwind ^bb1 : (!llvm.ptr) -> !llvm.ptr

// CHECK: [[BB4:.*]]:
// CHECK: invoke void (ptr, ...) @vararg_foo(ptr %[[a1]], i32 0)
^bb4:
  llvm.invoke @vararg_foo(%6, %0) to ^bb2 unwind ^bb1 vararg(!llvm.func<void (ptr, ...)>) : (!llvm.ptr, i32) -> ()

// CHECK: [[BB5:.*]]:
// CHECK: invoke void (ptr, ...) undef(ptr %[[a1]], i32 0)
^bb5:
  %9 = llvm.mlir.undef : !llvm.ptr
  llvm.invoke %9(%6, %0) to ^bb2 unwind ^bb1 vararg(!llvm.func<void (ptr, ...)>) : !llvm.ptr, (!llvm.ptr, i32) -> ()
}

// -----

llvm.func @foo() -> i8
llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: @invoke_result
// CHECK-SAME: %[[a0:[0-9]+]]
llvm.func @invoke_result(%arg0 : !llvm.ptr) attributes { personality = @__gxx_personality_v0 } {
// CHECK: %[[a1:[0-9]+]] = invoke i8 @foo()
// CHECK-NEXT: to label %[[normal:[0-9]+]] unwind label %[[unwind:[0-9]+]]
    %0 = llvm.invoke @foo() to ^bb1 unwind ^bb2 : () -> i8

// CHECK: [[normal]]:
// CHECK-NEXT: store i8 %[[a1]], ptr %[[a0]]
// CHECK-NEXT: ret void
^bb1:
    llvm.store %0, %arg0 : i8, !llvm.ptr
    llvm.return

// CHECK: [[unwind]]:
// CHECK-NEXT: landingpad { ptr, i32 }
// CHECK-NEXT: cleanup
// CHECK-NEXT: ret void
^bb2:
    %7 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    llvm.return
}

// -----

llvm.func @foo()
llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: @invoke_phis
llvm.func @invoke_phis() -> i32 attributes { personality = @__gxx_personality_v0 } {
// CHECK: invoke void @foo()
// CHECK-NEXT: to label %[[normal:[0-9]+]] unwind label %[[unwind:[0-9]+]]
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.invoke @foo() to ^bb1(%0 : i32) unwind ^bb2 : () -> ()

// CHECK: [[normal]]:
// CHECK-NEXT: %[[a1:[0-9]+]] = phi i32 [ 1, %[[unwind]] ], [ 0, %0 ]
// CHECK-NEXT: ret i32 %[[a1]]
^bb1(%1 : i32):
    llvm.return %1 : i32

// CHECK: [[unwind]]:
// CHECK-NEXT: landingpad { ptr, i32 }
// CHECK-NEXT: cleanup
// CHECK-NEXT: br label %[[normal]]
^bb2:
    %2 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
    %3 = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^bb1(%3 : i32)
}

// -----

// CHECK-LABEL: @hasGCFunction
// CHECK-SAME: gc "statepoint-example"
llvm.func @hasGCFunction() attributes { garbageCollector = "statepoint-example" } {
    llvm.return
}

// -----

// CHECK-LABEL: @gc_decl
// CHECK-SAME: gc "statepoint-example"
llvm.func @gc_decl() attributes { garbageCollector = "statepoint-example" }

// -----

// CHECK-LABEL: @section_func
// CHECK-SAME: section ".section.name"
llvm.func @section_func() attributes { section = ".section.name" } {
    llvm.return
}

// -----

// CHECK-LABEL: @local_unnamed_addr_func
// CHECK-SAME: local_unnamed_addr
llvm.func local_unnamed_addr @local_unnamed_addr_func() {
    llvm.return
}

// -----

// CHECK-LABEL: @unnamed_addr_func
// CHECK-SAME: unnamed_addr
llvm.func unnamed_addr @unnamed_addr_func()

// -----

// CHECK-LABEL: @align_func
// CHECK-SAME: align 2
llvm.func @align_func() attributes {alignment = 2 : i64} {
    llvm.return
}

// -----

// CHECK-LABEL: @align_decl
// CHECK-SAME: align 64
llvm.func @align_decl() attributes {alignment = 64 : i64}

// -----

// CHECK-LABEL: @callFreezeOp
llvm.func @callFreezeOp(%x : i32) {
  // CHECK: freeze i32 %{{[0-9]+}}
  %0 = llvm.freeze %x : i32
  %1 = llvm.mlir.undef : i32
  // CHECK: freeze i32 undef
  %2 = llvm.freeze %1 : i32
  %3 = llvm.mlir.poison : i32
  // CHECK: freeze i32 poison
  %4 = llvm.freeze %3 : i32
  llvm.return
}

// CHECK-LABEL: @freezeUsed
llvm.func @freezeUsed(%x : i32) -> i64 {
  // CHECK: %[[frozen:.*]] = freeze i32
  %frozen = llvm.freeze %x : i32
  // CHECK: %[[ext:.*]] = sext i32 %[[frozen]] to i64
  %ext = llvm.sext %frozen : i32 to i64
  // CHECK: ret i64 %[[ext]]
  llvm.return %ext : i64
}

// CHECK-LABEL: @boolConstArg
llvm.func @boolConstArg() -> i1 {
  // CHECK: ret i1 false
  %0 = llvm.mlir.constant(true) : i1
  %1 = llvm.mlir.constant(false) : i1
  %2 = llvm.and %0, %1 : i1
  llvm.return %2 : i1
}

// CHECK-LABEL: @callFenceInst
llvm.func @callFenceInst() {
  // CHECK: fence syncscope("agent") release
  llvm.fence syncscope("agent") release
  // CHECK: fence release
  llvm.fence release
  // CHECK: fence release
  llvm.fence syncscope("") release
  llvm.return
}

// CHECK-LABEL: @passthrough
// CHECK: #[[ATTR_GROUP:[0-9]*]]
llvm.func @passthrough() attributes {passthrough = [["alignstack", "4"], "null_pointer_is_valid", ["foo", "bar"]]} {
  llvm.return
}

// CHECK: attributes #[[ATTR_GROUP]] = {
// CHECK-DAG: alignstack=4
// CHECK-DAG: null_pointer_is_valid
// CHECK-DAG: "foo"="bar"

// -----

// CHECK-LABEL: @my_allocator
// CHECK: #[[ALLOC_ATTRS:[0-9]*]]
llvm.func @my_allocator(i64) attributes {passthrough = [["allocsize", "4294967295"]]}

// CHECK: attributes #[[ALLOC_ATTRS]] = {
// CHECK-DAG: allocsize(0)

// -----

// CHECK-LABEL: @functionEntryCount
// CHECK-SAME: !prof ![[PROF_ID:[0-9]*]]
llvm.func @functionEntryCount() attributes {function_entry_count = 4242 : i64} {
  llvm.return
}

// CHECK: ![[PROF_ID]] = !{!"function_entry_count", i64 4242}

// -----

// CHECK-LABEL: @constant_bf16
llvm.func @constant_bf16() -> bf16 {
  %0 = llvm.mlir.constant(1.000000e+01 : bf16) : bf16
  llvm.return %0 : bf16
}

// CHECK: ret bfloat 0xR4120

// -----

llvm.func @address_taken() {
  llvm.return
}

llvm.mlir.global internal constant @taker_of_address() : !llvm.ptr {
  %0 = llvm.mlir.addressof @address_taken : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

// -----

// CHECK: @forward_use_of_address = linkonce global ptr @address_declared_after_use
llvm.mlir.global linkonce @forward_use_of_address() : !llvm.ptr {
  %0 = llvm.mlir.addressof @address_declared_after_use : !llvm.ptr
  llvm.return %0 : !llvm.ptr
}

llvm.mlir.global linkonce @address_declared_after_use() : f32

// -----

// CHECK: @take_self_address = linkonce global { i32, ptr } {{.*}} ptr @take_self_address
llvm.mlir.global linkonce @take_self_address() : !llvm.struct<(i32, !llvm.ptr)> {
  %z32 = llvm.mlir.constant(0 : i32) : i32
  %0 = llvm.mlir.undef : !llvm.struct<(i32, !llvm.ptr)>
  %1 = llvm.mlir.addressof @take_self_address : !llvm.ptr
  %2 = llvm.getelementptr %1[%z32, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32, !llvm.ptr)>
  %3 = llvm.insertvalue %z32, %0[0] : !llvm.struct<(i32, !llvm.ptr)>
  %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<(i32, !llvm.ptr)>
  llvm.return %4 : !llvm.struct<(i32, !llvm.ptr)>
}

// -----

// CHECK: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @foo, ptr null }]
llvm.mlir.global_ctors { ctors = [@foo], priorities = [0 : i32]}

llvm.func @foo() {
  llvm.return
}

// -----

// CHECK: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @foo, ptr null }]
llvm.mlir.global_dtors { dtors = [@foo], priorities = [0 : i32]}

llvm.func @foo() {
  llvm.return
}

// -----

// Check that branch weight attributes are exported properly as metadata.
llvm.func @cond_br_weights(%cond : i1, %arg0 : i32,  %arg1 : i32) -> i32 {
  // CHECK: !prof ![[NODE:[0-9]+]]
  llvm.cond_br %cond weights([5, 10]), ^bb1, ^bb2
^bb1:  // pred: ^bb0
  llvm.return %arg0 : i32
^bb2:  // pred: ^bb0
  llvm.return %arg1 : i32
}

// CHECK: ![[NODE]] = !{!"branch_weights", i32 5, i32 10}

// -----

llvm.func @fn()

// CHECK-LABEL: @call_branch_weights
llvm.func @call_branch_weights() {
  // CHECK: !prof ![[NODE:[0-9]+]]
  llvm.call @fn() {branch_weights = array<i32 : 42>} : () -> ()
  llvm.return
}

// CHECK: ![[NODE]] = !{!"branch_weights", i32 42}

// -----

llvm.func @fn() -> i32

// CHECK-LABEL: @call_branch_weights
llvm.func @call_branch_weights() {
  // CHECK: !prof ![[NODE:[0-9]+]]
  %res = llvm.call @fn() {branch_weights = array<i32 : 42>} : () -> i32
  llvm.return
}

// CHECK: ![[NODE]] = !{!"branch_weights", i32 42}

// -----

llvm.func @foo()
llvm.func @__gxx_personality_v0(...) -> i32

// CHECK-LABEL: @invoke_branch_weights
llvm.func @invoke_branch_weights() -> i32 attributes {personality = @__gxx_personality_v0} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: !prof ![[NODE:[0-9]+]]
  llvm.invoke @foo() to ^bb2 unwind ^bb1 {branch_weights = array<i32 : 42, 99>} : () -> ()
^bb1:  // pred: ^bb0
  %1 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  llvm.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  llvm.return %0 : i32
}

// CHECK: ![[NODE]] = !{!"branch_weights", i32 42, i32 99}

// -----

llvm.func @volatile_store_and_load() {
  %val = llvm.mlir.constant(5 : i32) : i32
  %size = llvm.mlir.constant(1 : i64) : i64
  %0 = llvm.alloca %size x i32 : (i64) -> (!llvm.ptr)
  // CHECK: store volatile i32 5, ptr %{{.*}}
  llvm.store volatile %val, %0 : i32, !llvm.ptr
  // CHECK: %{{.*}} = load volatile i32, ptr %{{.*}}
  %1 = llvm.load volatile %0: !llvm.ptr -> i32
  llvm.return
}

// -----

// Check that nontemporal attribute is exported as metadata node.
llvm.func @nontemporal_store_and_load() {
  %val = llvm.mlir.constant(5 : i32) : i32
  %size = llvm.mlir.constant(1 : i64) : i64
  %0 = llvm.alloca %size x i32 : (i64) -> (!llvm.ptr)
  // CHECK: !nontemporal ![[NODE:[0-9]+]]
  llvm.store %val, %0 {nontemporal} : i32, !llvm.ptr
  // CHECK: !nontemporal ![[NODE]]
  %1 = llvm.load %0 {nontemporal} : !llvm.ptr -> i32
  llvm.return
}

// CHECK: ![[NODE]] = !{i32 1}

// -----

// Check that invariantLoad attribute is exported as metadata node.
llvm.func @nontemporal_store_and_load(%ptr : !llvm.ptr) -> i32 {
  // CHECK: !invariant.load ![[NODE:[0-9]+]]
  %1 = llvm.load %ptr invariant : !llvm.ptr -> i32
  llvm.return %1 : i32
}

// CHECK: ![[NODE]] = !{}

// -----

llvm.func @atomic_store_and_load(%ptr : !llvm.ptr) {
  // CHECK: load atomic
  // CHECK-SAME:  acquire, align 4
  %1 = llvm.load %ptr atomic acquire {alignment = 4 : i64} : !llvm.ptr -> f32
  // CHECK: load atomic
  // CHECK-SAME:  syncscope("singlethread") acquire, align 4
  %2 = llvm.load %ptr atomic syncscope("singlethread") acquire {alignment = 4 : i64} : !llvm.ptr -> f32

  // CHECK: store atomic
  // CHECK-SAME:  release, align 4
  llvm.store %1, %ptr atomic release {alignment = 4 : i64} : f32, !llvm.ptr
  // CHECK: store atomic
  // CHECK-SAME:  syncscope("singlethread") release, align 4
  llvm.store %2, %ptr atomic syncscope("singlethread") release {alignment = 4 : i64} : f32, !llvm.ptr
  llvm.return
}

// -----

// Check that the translation does not crash in absence of a data layout.
module {
  // CHECK: declare void @module_default_layout
  llvm.func @module_default_layout()
}

// -----

// CHECK: target datalayout = "E"
module attributes {llvm.data_layout = "E"} {
  llvm.func @module_big_endian()
}

// -----

// CHECK: "CodeView", i32 1
module attributes {llvm.target_triple = "x86_64-pc-windows-msvc"} {}

// -----

// CHECK-NOT: "CodeView", i32 1
// CHECK: aarch64-linux-android
module attributes {llvm.target_triple = "aarch64-linux-android"} {}

// -----

// CHECK-NOT: "CodeView", i32 1
module attributes {} {}

// -----

// CHECK-LABEL: @useInlineAsm
llvm.func @useInlineAsm(%arg0: i32) {
  // Constraints string is checked at LLVM InlineAsm instruction construction time.
  // So we can't just use "bar" everywhere, number of in/out arguments has to match.

  // CHECK-NEXT:  call void asm "foo", "r"(i32 {{.*}})
  llvm.inline_asm "foo", "r" %arg0 : (i32) -> ()

  // CHECK-NEXT:  call i8 asm "foo", "=r,r"(i32 {{.*}})
  %0 = llvm.inline_asm "foo", "=r,r" %arg0 : (i32) -> i8

  // CHECK-NEXT:  call i8 asm "foo", "=r,r,r"(i32 {{.*}}, i32 {{.*}})
  %1 = llvm.inline_asm "foo", "=r,r,r" %arg0, %arg0 : (i32, i32) -> i8

  // CHECK-NEXT:  call i8 asm sideeffect "foo", "=r,r,r"(i32 {{.*}}, i32 {{.*}})
  %2 = llvm.inline_asm has_side_effects "foo", "=r,r,r" %arg0, %arg0 : (i32, i32) -> i8

  // CHECK-NEXT:  call i8 asm alignstack "foo", "=r,r,r"(i32 {{.*}}, i32 {{.*}})
  %3 = llvm.inline_asm is_align_stack "foo", "=r,r,r" %arg0, %arg0 : (i32, i32) -> i8

  // CHECK-NEXT:  call i8 asm inteldialect "foo", "=r,r,r"(i32 {{.*}}, i32 {{.*}})
  %4 = llvm.inline_asm asm_dialect = "intel" "foo", "=r,r,r" %arg0, %arg0 : (i32, i32) -> i8

  // CHECK-NEXT:  call { i8, i8 } asm "foo", "=r,=r,r"(i32 {{.*}})
  %5 = llvm.inline_asm "foo", "=r,=r,r" %arg0 : (i32) -> !llvm.struct<(i8, i8)>

  llvm.return
}

// -----

llvm.func @fastmathFlagsFunc(f32) -> f32

// CHECK-LABEL: @fastmathFlags
llvm.func @fastmathFlags(%arg0: f32, %arg1 : vector<2xf32>) {
// CHECK: {{.*}} = fadd nnan ninf float {{.*}}, {{.*}}
// CHECK: {{.*}} = fsub nnan ninf float {{.*}}, {{.*}}
// CHECK: {{.*}} = fmul nnan ninf float {{.*}}, {{.*}}
// CHECK: {{.*}} = fdiv nnan ninf float {{.*}}, {{.*}}
// CHECK: {{.*}} = frem nnan ninf float {{.*}}, {{.*}}
  %0 = llvm.fadd %arg0, %arg0 {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32
  %1 = llvm.fsub %arg0, %arg0 {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32
  %2 = llvm.fmul %arg0, %arg0 {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32
  %3 = llvm.fdiv %arg0, %arg0 {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32
  %4 = llvm.frem %arg0, %arg0 {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32

// CHECK: {{.*}} = fcmp nnan ninf oeq {{.*}}, {{.*}}
  %5 = llvm.fcmp "oeq" %arg0, %arg0 {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32

// CHECK: {{.*}} = fneg nnan ninf float {{.*}}
  %6 = llvm.fneg %arg0 {fastmathFlags = #llvm.fastmath<nnan, ninf>} : f32

// CHECK: {{.*}} = call float @fastmathFlagsFunc({{.*}})
// CHECK: {{.*}} = call nnan float @fastmathFlagsFunc({{.*}})
// CHECK: {{.*}} = call ninf float @fastmathFlagsFunc({{.*}})
// CHECK: {{.*}} = call nsz float @fastmathFlagsFunc({{.*}})
// CHECK: {{.*}} = call arcp float @fastmathFlagsFunc({{.*}})
// CHECK: {{.*}} = call contract float @fastmathFlagsFunc({{.*}})
// CHECK: {{.*}} = call afn float @fastmathFlagsFunc({{.*}})
// CHECK: {{.*}} = call reassoc float @fastmathFlagsFunc({{.*}})
// CHECK: {{.*}} = call fast float @fastmathFlagsFunc({{.*}})
  %8 = llvm.call @fastmathFlagsFunc(%arg0) {fastmathFlags = #llvm.fastmath<none>} : (f32) -> (f32)
  %9 = llvm.call @fastmathFlagsFunc(%arg0) {fastmathFlags = #llvm.fastmath<nnan>} : (f32) -> (f32)
  %10 = llvm.call @fastmathFlagsFunc(%arg0) {fastmathFlags = #llvm.fastmath<ninf>} : (f32) -> (f32)
  %11 = llvm.call @fastmathFlagsFunc(%arg0) {fastmathFlags = #llvm.fastmath<nsz>} : (f32) -> (f32)
  %12 = llvm.call @fastmathFlagsFunc(%arg0) {fastmathFlags = #llvm.fastmath<arcp>} : (f32) -> (f32)
  %13 = llvm.call @fastmathFlagsFunc(%arg0) {fastmathFlags = #llvm.fastmath<contract>} : (f32) -> (f32)
  %14 = llvm.call @fastmathFlagsFunc(%arg0) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> (f32)
  %15 = llvm.call @fastmathFlagsFunc(%arg0) {fastmathFlags = #llvm.fastmath<reassoc>} : (f32) -> (f32)
  %16 = llvm.call @fastmathFlagsFunc(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> (f32)

// CHECK: call fast float @llvm.copysign.f32(float {{.*}}, float {{.*}})
  %17 = "llvm.intr.copysign"(%arg0, %arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f32, f32) -> f32
// CHECK: call afn float @llvm.copysign.f32(float {{.*}}, float {{.*}})
  %18 = "llvm.intr.copysign"(%arg0, %arg0) {fastmathFlags = #llvm.fastmath<afn>} : (f32, f32) -> f32

// CHECK: call fast float @llvm.powi.f32.i32(float {{.*}}, i32 {{.*}})
  %exp = llvm.mlir.constant(1 : i32) : i32
  %19 = "llvm.intr.powi"(%arg0, %exp) {fastmathFlags = #llvm.fastmath<fast>} : (f32, i32) -> f32
// CHECK: call afn float @llvm.powi.f32.i32(float {{.*}}, i32 {{.*}})
  %20 = "llvm.intr.powi"(%arg0, %exp) {fastmathFlags = #llvm.fastmath<afn>} : (f32, i32) -> f32

// CHECK: call nnan float @llvm.vector.reduce.fmax.v2f32(<2 x float> {{.*}})
// CHECK: call nnan float @llvm.vector.reduce.fmin.v2f32(<2 x float> {{.*}})
  %21 = llvm.intr.vector.reduce.fmax(%arg1) {fastmathFlags = #llvm.fastmath<nnan>} : (vector<2xf32>) -> f32
  %22 = llvm.intr.vector.reduce.fmin(%arg1) {fastmathFlags = #llvm.fastmath<nnan>} : (vector<2xf32>) -> f32

// CHECK: call nnan float @llvm.vector.reduce.fmaximum.v2f32(<2 x float> {{.*}})
// CHECK: call nnan float @llvm.vector.reduce.fminimum.v2f32(<2 x float> {{.*}})
  %23 = llvm.intr.vector.reduce.fmaximum(%arg1) {fastmathFlags = #llvm.fastmath<nnan>} : (vector<2xf32>) -> f32
  %24 = llvm.intr.vector.reduce.fminimum(%arg1) {fastmathFlags = #llvm.fastmath<nnan>} : (vector<2xf32>) -> f32

  %25 = llvm.mlir.constant(true) : i1
// CHECK: select contract i1
  %26 = llvm.select %25, %arg0, %20 {fastmathFlags = #llvm.fastmath<contract>} : i1, f32
  llvm.return
}

// -----

// CHECK-LABEL: @switch_empty
llvm.func @switch_empty(%arg0 : i32) -> i32 {
  // CHECK:      switch i32 %[[SWITCH_arg0:[0-9]+]], label %[[SWITCHDEFAULT_bb1:[0-9]+]] [
  // CHECK-NEXT: ]
  llvm.switch %arg0 : i32, ^bb1 [
  ]

  // CHECK:      [[SWITCHDEFAULT_bb1]]:
  // CHECK-NEXT:   ret i32 %[[SWITCH_arg0]]
^bb1:
  llvm.return %arg0 : i32
}

// -----

// CHECK-LABEL: @switch_args
llvm.func @switch_args(%arg0: i32) -> i32 {
  %0 = llvm.mlir.constant(5 : i32) : i32
  %1 = llvm.mlir.constant(7 : i32) : i32
  %2 = llvm.mlir.constant(11 : i32) : i32
  // CHECK:      switch i32 %[[SWITCH_arg0:[0-9]+]], label %[[SWITCHDEFAULT_bb1:[0-9]+]] [
  // CHECK-NEXT:   i32 -1, label %[[SWITCHCASE_bb2:[0-9]+]]
  // CHECK-NEXT:   i32 1, label %[[SWITCHCASE_bb3:[0-9]+]]
  // CHECK-NEXT: ]
  llvm.switch %arg0 : i32, ^bb1 [
    -1: ^bb2(%0 : i32),
    1: ^bb3(%1, %2 : i32, i32)
  ]

// CHECK:      [[SWITCHDEFAULT_bb1]]:
// CHECK-NEXT:   ret i32 %[[SWITCH_arg0]]
^bb1:  // pred: ^bb0
  llvm.return %arg0 : i32

// CHECK:      [[SWITCHCASE_bb2]]:
// CHECK-NEXT:   phi i32 [ 5, %1 ]
// CHECK-NEXT:   ret i32
^bb2(%3: i32): // pred: ^bb0
  llvm.return %1 : i32

// CHECK:      [[SWITCHCASE_bb3]]:
// CHECK-NEXT:   phi i32 [ 7, %1 ]
// CHECK-NEXT:   phi i32 [ 11, %1 ]
// CHECK-NEXT:   ret i32
^bb3(%4: i32, %5: i32): // pred: ^bb0
  llvm.return %4 : i32
}

// CHECK-LABEL: @switch_weights
llvm.func @switch_weights(%arg0: i32) -> i32 {
  %0 = llvm.mlir.constant(19 : i32) : i32
  %1 = llvm.mlir.constant(23 : i32) : i32
  %2 = llvm.mlir.constant(29 : i32) : i32
  // CHECK: !prof ![[SWITCH_WEIGHT_NODE:[0-9]+]]
  llvm.switch %arg0 : i32, ^bb1(%0 : i32) [
    9: ^bb2(%1, %2 : i32, i32),
    99: ^bb3
  ] {branch_weights = array<i32 : 13, 17, 19>}

^bb1(%3: i32):  // pred: ^bb0
  llvm.return %3 : i32

^bb2(%4: i32, %5: i32): // pred: ^bb0
  llvm.return %5 : i32

^bb3: // pred: ^bb0
  llvm.return %arg0 : i32
}

// CHECK: ![[SWITCH_WEIGHT_NODE]] = !{!"branch_weights", i32 13, i32 17, i32 19}

// -----

// It is okay to have repeated successors if they have no arguments.

// CHECK-LABEL: @duplicate_block_in_switch
// CHECK-SAME: float %[[FIRST:.*]],
// CHECK-SAME: float %[[SECOND:.*]])
// CHECK:   switch i32 %{{.*}}, label %[[DEFAULT:.*]] [
// CHECK:     i32 105, label %[[DUPLICATE:.*]]
// CHECK:     i32 108, label %[[BLOCK:.*]]
// CHECK:     i32 106, label %[[DUPLICATE]]
// CHECK:   ]

// CHECK: [[DEFAULT]]:
// CHECK:   phi float [ %[[FIRST]], %{{.*}} ]
// CHECK:   call void @bar

// CHECK: [[DUPLICATE]]:
// CHECK:   call void @baz

// CHECK: [[BLOCK]]:
// CHECK:   phi float [ %[[SECOND]], %{{.*}} ]
// CHECK:   call void @qux

llvm.func @duplicate_block_in_switch(%cond : i32, %arg1: f32, %arg2: f32) {
  llvm.switch %cond : i32, ^bb1(%arg1: f32) [
    105: ^bb2,
    108: ^bb3(%arg2: f32),
    106: ^bb2
  ]

^bb1(%arg3: f32):
  llvm.call @bar(%arg3): (f32) -> ()
  llvm.return

^bb2:
  llvm.call @baz() : () -> ()
  llvm.return

^bb3(%arg4: f32):
  llvm.call @qux(%arg4) : (f32) -> ()
  llvm.return
}

// If there are repeated successors with arguments, a new block must be created
// for repeated successors to ensure PHI can disambiguate values based on the
// predecessor they come from.

// CHECK-LABEL: @duplicate_block_with_args_in_switch
// CHECK-SAME: float %[[FIRST:.*]],
// CHECK-SAME: float %[[SECOND:.*]])
// CHECK:   switch i32 %{{.*}}, label %[[DEFAULT:.*]] [
// CHECK:     i32 106, label %[[DUPLICATE:.*]]
// CHECK:     i32 105, label %[[BLOCK:.*]]
// CHECK:     i32 108, label %[[DEDUPLICATED:.*]]
// CHECK:   ]

// CHECK: [[DEFAULT]]:
// CHECK:   phi float [ %[[FIRST]], %{{.*}} ]
// CHECK:   call void @bar

// CHECK: [[BLOCK]]:
// CHECK:   call void @baz

// CHECK: [[DUPLICATE]]:
// CHECK:   phi float [ %[[PHI:.*]], %[[DEDUPLICATED]] ], [ %[[FIRST]], %{{.*}} ]
// CHECK:   call void @qux

// CHECK: [[DEDUPLICATED]]:
// CHECK:   %[[PHI]] = phi float [ %[[SECOND]], %{{.*}} ]
// CHECK:   br label %[[DUPLICATE]]

llvm.func @duplicate_block_with_args_in_switch(%cond : i32, %arg1: f32, %arg2: f32) {
  llvm.switch %cond : i32, ^bb1(%arg1: f32) [
    106: ^bb3(%arg1: f32),
    105: ^bb2,
    108: ^bb3(%arg2: f32)
  ]

^bb1(%arg3: f32):
  llvm.call @bar(%arg3): (f32) -> ()
  llvm.return

^bb2:
  llvm.call @baz() : () -> ()
  llvm.return

^bb3(%arg4: f32):
  llvm.call @qux(%arg4) : (f32) -> ()
  llvm.return
}

llvm.func @bar(f32)
llvm.func @baz()
llvm.func @qux(f32)

// -----

// Varaidic function definition

// CHECK: %struct.va_list = type { ptr }

// CHECK: define void @vararg_function(i32 %{{.*}}, ...)
llvm.func @vararg_function(%arg0: i32, ...) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA0:.+]] = alloca %struct.va_list, align 8
  %2 = llvm.alloca %1 x !llvm.struct<"struct.va_list", (ptr)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK: call void @llvm.va_start.p0(ptr %[[ALLOCA0]])
  llvm.intr.vastart %2 : !llvm.ptr
  // CHECK: %[[ALLOCA1:.+]] = alloca ptr, align 8
  %4 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK: call void @llvm.va_copy.p0(ptr %[[ALLOCA1]], ptr %[[ALLOCA0]])
  llvm.intr.vacopy %2 to %4 : !llvm.ptr, !llvm.ptr
  // CHECK: call void @llvm.va_end.p0(ptr %[[ALLOCA1]])
  // CHECK: call void @llvm.va_end.p0(ptr %[[ALLOCA0]])
  llvm.intr.vaend %4 : !llvm.ptr
  llvm.intr.vaend %2 : !llvm.ptr
  // CHECK: ret void
  llvm.return
}

// -----

// CHECK: declare void @readonly_function([[PTR:.+]] readonly)
llvm.func @readonly_function(%arg0: !llvm.ptr {llvm.readonly})

// -----

// CHECK: declare void @arg_mem_none_func() #[[ATTR:[0-9]+]]
llvm.func @arg_mem_none_func() attributes {
  memory = #llvm.memory_effects<other = readwrite, argMem = none, inaccessibleMem = readwrite>}

// CHECK: attributes #[[ATTR]] = { memory(readwrite, argmem: none) }

// -----

// CHECK: declare void @readwrite_func() #[[ATTR:[0-9]+]]
llvm.func @readwrite_func() attributes {
  memory = #llvm.memory_effects<other = readwrite, argMem = readwrite, inaccessibleMem = readwrite>}

// CHECK: attributes #[[ATTR]] = { memory(readwrite) }

// -----

//
// arm_streaming attribute.
//

// CHECK-LABEL: @streaming_func
// CHECK: #[[ATTR:[0-9]*]]
llvm.func @streaming_func() attributes {arm_streaming} {
  llvm.return
}

// CHECK: attributes #[[ATTR]] = { "aarch64_pstate_sm_enabled" }

// -----

//
// arm_locally_streaming attribute.
//

// CHECK-LABEL: @locally_streaming_func
// CHECK: #[[ATTR:[0-9]*]]
llvm.func @locally_streaming_func() attributes {arm_locally_streaming} {
  llvm.return
}

// CHECK: attributes #[[ATTR]] = { "aarch64_pstate_sm_body" }

// -----

//
// arm_streaming_compatible attribute.
//

// CHECK-LABEL: @streaming_compatible_func
// CHECK: #[[ATTR:[0-9]*]]
llvm.func @streaming_compatible_func() attributes {arm_streaming_compatible} {
  llvm.return
}

// CHECK: attributes #[[ATTR]] = { "aarch64_pstate_sm_compatible" }

// -----

// CHECK-LABEL: @new_za_func
// CHECK-SAME: #[[ATTR:[0-9]*]]
llvm.func @new_za_func() attributes {arm_new_za} {
  llvm.return
}
// CHECK: #[[ATTR]] = { "aarch64_new_za" }

// -----

// CHECK-LABEL: @in_za_func
// CHECK-SAME: #[[ATTR:[0-9]*]]
llvm.func @in_za_func() attributes {arm_in_za } {
  llvm.return
}
// CHECK: #[[ATTR]] = { "aarch64_in_za" }

// -----

// CHECK-LABEL: @out_za_func
// CHECK-SAME: #[[ATTR:[0-9]*]]
llvm.func @out_za_func() attributes {arm_out_za } {
  llvm.return
}
// CHECK: #[[ATTR]] = { "aarch64_out_za" }

// -----

// CHECK-LABEL: @inout_za_func
// CHECK-SAME: #[[ATTR:[0-9]*]]
llvm.func @inout_za_func() attributes {arm_inout_za } {
  llvm.return
}
// CHECK: #[[ATTR]] = { "aarch64_inout_za" }

// -----

// CHECK-LABEL: @preserves_za_func
// CHECK-SAME: #[[ATTR:[0-9]*]]
llvm.func @preserves_za_func() attributes {arm_preserves_za} {
  llvm.return
}
// CHECK: #[[ATTR]] = { "aarch64_preserves_za" }

// -----

//
// Zero-initialize operation.
//

// CHECK: @partially_zeroinit_aggregate = linkonce global { i32, i64, [3 x i8] } { i32 0, i64 1, [3 x i8] zeroinitializer }
llvm.mlir.global linkonce @partially_zeroinit_aggregate() : !llvm.struct<(i32, i64, !llvm.array<3 x i8>)> {
  %0 = llvm.mlir.zero : !llvm.struct<(i32, i64, !llvm.array<3 x i8>)>
  %1 = llvm.mlir.constant(1 : i64) : i64
  %2 = llvm.insertvalue %1, %0[1] : !llvm.struct<(i32, i64, !llvm.array<3 x i8>)>
  llvm.return %2 : !llvm.struct<(i32, i64, !llvm.array<3 x i8>)>
}

llvm.func @zeroinit_complex_local_aggregate() {
  // CHECK: %[[#VAR:]] = alloca [1000 x { i32, [3 x { double, <4 x ptr>, [2 x ptr] }], [6 x ptr] }], i64 1, align 32
  %0 = llvm.mlir.constant(1 : i64) : i64
  %1 = llvm.alloca %0 x !llvm.array<1000 x !llvm.struct<(i32, !llvm.array<3 x !llvm.struct<(f64, !llvm.vec<4 x ptr>, !llvm.array<2 x ptr>)>>, !llvm.array<6 x ptr>)>> : (i64) -> !llvm.ptr

  // CHECK: store [1000 x { i32, [3 x { double, <4 x ptr>, [2 x ptr] }], [6 x ptr] }] zeroinitializer, ptr %[[#VAR]], align 32
  %2 = llvm.mlir.zero : !llvm.array<1000 x !llvm.struct<(i32, !llvm.array<3 x !llvm.struct<(f64, !llvm.vec<4 x ptr>, !llvm.array<2 x ptr>)>>, !llvm.array<6 x ptr>)>>
  llvm.store %2, %1 : !llvm.array<1000 x !llvm.struct<(i32, !llvm.array<3 x !llvm.struct<(f64, !llvm.vec<4 x ptr>, !llvm.array<2 x ptr>)>>, !llvm.array<6 x ptr>)>>, !llvm.ptr

  llvm.return
}

//CHECK: !llvm.linker.options = !{![[MD0:[0-9]+]], ![[MD1:[0-9]+]]}
//CHECK: ![[MD0]] = !{!"/DEFAULTLIB:", !"libcmt"}
llvm.linker_options ["/DEFAULTLIB:", "libcmt"]
//CHECK: ![[MD1]] = !{!"/DEFAULTLIB:", !"libcmtd"}
llvm.linker_options ["/DEFAULTLIB:", "libcmtd"]

// -----

// CHECK: @big_ = common global [4294967296 x i8] zeroinitializer
llvm.mlir.global common @big_(dense<0> : vector<4294967296xi8>) {addr_space = 0 : i32} : !llvm.array<4294967296 x i8>

// -----

// CHECK-LABEL: @no_inline
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @no_inline() attributes { no_inline } {
  llvm.return
}

// CHECK: #[[ATTRS]]
// CHECK-SAME: noinline

// -----

// CHECK-LABEL: @always_inline
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @always_inline() attributes { always_inline } {
  llvm.return
}

// CHECK: #[[ATTRS]]
// CHECK-SAME: alwaysinline

// -----

// CHECK-LABEL: @optimize_none
// CHECK-SAME: #[[ATTRS:[0-9]+]]
llvm.func @optimize_none() attributes { no_inline, optimize_none } {
  llvm.return
}

// CHECK: #[[ATTRS]]
// CHECK-SAME: optnone
