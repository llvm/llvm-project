// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

// CHECK: @i32_global = internal global i32 42
llvm.mlir.global internal @i32_global(42: i32) : !llvm.i32

// CHECK: @i32_const = internal constant i53 52
llvm.mlir.global internal constant @i32_const(52: i53) : !llvm.i53

// CHECK: @int_global_array = internal global [3 x i32] [i32 62, i32 62, i32 62]
llvm.mlir.global internal @int_global_array(dense<62> : vector<3xi32>) : !llvm.array<3 x i32>

// CHECK: @i32_global_addr_space = internal addrspace(7) global i32 62
llvm.mlir.global internal @i32_global_addr_space(62: i32) {addr_space = 7 : i32} : !llvm.i32

// CHECK: @float_global = internal global float 0.000000e+00
llvm.mlir.global internal @float_global(0.0: f32) : !llvm.float

// CHECK: @float_global_array = internal global [1 x float] [float -5.000000e+00]
llvm.mlir.global internal @float_global_array(dense<[-5.0]> : vector<1xf32>) : !llvm.array<1 x float>

// CHECK: @string_const = internal constant [6 x i8] c"foobar"
llvm.mlir.global internal constant @string_const("foobar") : !llvm.array<6 x i8>

// CHECK: @int_global_undef = internal global i64 undef
llvm.mlir.global internal @int_global_undef() : !llvm.i64

// CHECK: @int_gep = internal constant i32* getelementptr (i32, i32* @i32_global, i32 2)
llvm.mlir.global internal constant @int_gep() : !llvm.ptr<i32> {
  %addr = llvm.mlir.addressof @i32_global : !llvm.ptr<i32>
  %_c0 = llvm.mlir.constant(2: i32) :!llvm.i32
  %gepinit = llvm.getelementptr %addr[%_c0] : (!llvm.ptr<i32>, !llvm.i32) -> !llvm.ptr<i32>
  llvm.return %gepinit : !llvm.ptr<i32>
}

//
// Linkage attribute.
//

// CHECK: @private = private global i32 42
llvm.mlir.global private @private(42 : i32) : !llvm.i32
// CHECK: @internal = internal global i32 42
llvm.mlir.global internal @internal(42 : i32) : !llvm.i32
// CHECK: @available_externally = available_externally global i32 42
llvm.mlir.global available_externally @available_externally(42 : i32) : !llvm.i32
// CHECK: @linkonce = linkonce global i32 42
llvm.mlir.global linkonce @linkonce(42 : i32) : !llvm.i32
// CHECK: @weak = weak global i32 42
llvm.mlir.global weak @weak(42 : i32) : !llvm.i32
// CHECK: @common = common global i32 42
llvm.mlir.global common @common(42 : i32) : !llvm.i32
// CHECK: @appending = appending global i32 42
llvm.mlir.global appending @appending(42 : i32) : !llvm.i32
// CHECK: @extern_weak = extern_weak global i32
llvm.mlir.global extern_weak @extern_weak() : !llvm.i32
// CHECK: @linkonce_odr = linkonce_odr global i32 42
llvm.mlir.global linkonce_odr @linkonce_odr(42 : i32) : !llvm.i32
// CHECK: @weak_odr = weak_odr global i32 42
llvm.mlir.global weak_odr @weak_odr(42 : i32) : !llvm.i32
// CHECK: @external = external global i32
llvm.mlir.global external @external() : !llvm.i32


//
// Declarations of the allocation functions to be linked against. These are
// inserted before other functions in the module.
//

// CHECK: declare i8* @malloc(i64)
llvm.func @malloc(!llvm.i64) -> !llvm.ptr<i8>
// CHECK: declare void @free(i8*)


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
  // CHECK: load i32, i32* @i32_global
  %0 = llvm.mlir.addressof @i32_global : !llvm.ptr<i32>
  %1 = llvm.load %0 : !llvm.ptr<i32>

  // Check the contracted form of load from array constants.
  // CHECK: load i8, i8* getelementptr inbounds ([6 x i8], [6 x i8]* @string_const, i64 0, i64 0)
  %2 = llvm.mlir.addressof @string_const : !llvm.ptr<array<6 x i8>>
  %c0 = llvm.mlir.constant(0 : index) : !llvm.i64
  %3 = llvm.getelementptr %2[%c0, %c0] : (!llvm.ptr<array<6 x i8>>, !llvm.i64, !llvm.i64) -> !llvm.ptr<i8>
  %4 = llvm.load %3 : !llvm.ptr<i8>

  llvm.return
}

// CHECK-LABEL: declare void @body(i64)
llvm.func @body(!llvm.i64)


// CHECK-LABEL: define void @simple_loop()
llvm.func @simple_loop() {
// CHECK: br label %[[SIMPLE_bb1:[0-9]+]]
  llvm.br ^bb1

// Constants are inlined in LLVM rather than a separate instruction.
// CHECK: [[SIMPLE_bb1]]:
// CHECK-NEXT: br label %[[SIMPLE_bb2:[0-9]+]]
^bb1:   // pred: ^bb0
  %0 = llvm.mlir.constant(1 : index) : !llvm.i64
  %1 = llvm.mlir.constant(42 : index) : !llvm.i64
  llvm.br ^bb2(%0 : !llvm.i64)

// CHECK: [[SIMPLE_bb2]]:
// CHECK-NEXT:   %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %[[SIMPLE_bb3:[0-9]+]] ], [ 1, %[[SIMPLE_bb1]] ]
// CHECK-NEXT:   %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 42
// CHECK-NEXT:   br i1 %{{[0-9]+}}, label %[[SIMPLE_bb3]], label %[[SIMPLE_bb4:[0-9]+]]
^bb2(%2: !llvm.i64): // 2 preds: ^bb1, ^bb3
  %3 = llvm.icmp "slt" %2, %1 : !llvm.i64
  llvm.cond_br %3, ^bb3, ^bb4

// CHECK: [[SIMPLE_bb3]]:
// CHECK-NEXT:   call void @body(i64 %{{[0-9]+}})
// CHECK-NEXT:   %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
// CHECK-NEXT:   br label %[[SIMPLE_bb2]]
^bb3:   // pred: ^bb2
  llvm.call @body(%2) : (!llvm.i64) -> ()
  %4 = llvm.mlir.constant(1 : index) : !llvm.i64
  %5 = llvm.add %2, %4 : !llvm.i64
  llvm.br ^bb2(%5 : !llvm.i64)

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
llvm.func @body_args(!llvm.i64) -> !llvm.i64
// CHECK-LABEL: declare i32 @other(i64, i32)
llvm.func @other(!llvm.i64, !llvm.i32) -> !llvm.i32

// CHECK-LABEL: define i32 @func_args(i32 {{%.*}}, i32 {{%.*}})
// CHECK-NEXT: br label %[[ARGS_bb1:[0-9]+]]
llvm.func @func_args(%arg0: !llvm.i32, %arg1: !llvm.i32) -> !llvm.i32 {
  %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
  llvm.br ^bb1

// CHECK: [[ARGS_bb1]]:
// CHECK-NEXT: br label %[[ARGS_bb2:[0-9]+]]
^bb1:   // pred: ^bb0
  %1 = llvm.mlir.constant(0 : index) : !llvm.i64
  %2 = llvm.mlir.constant(42 : index) : !llvm.i64
  llvm.br ^bb2(%1 : !llvm.i64)

// CHECK: [[ARGS_bb2]]:
// CHECK-NEXT:   %5 = phi i64 [ %12, %[[ARGS_bb3:[0-9]+]] ], [ 0, %[[ARGS_bb1]] ]
// CHECK-NEXT:   %6 = icmp slt i64 %5, 42
// CHECK-NEXT:   br i1 %6, label %[[ARGS_bb3]], label %[[ARGS_bb4:[0-9]+]]
^bb2(%3: !llvm.i64): // 2 preds: ^bb1, ^bb3
  %4 = llvm.icmp "slt" %3, %2 : !llvm.i64
  llvm.cond_br %4, ^bb3, ^bb4

// CHECK: [[ARGS_bb3]]:
// CHECK-NEXT:   %8 = call i64 @body_args(i64 %5)
// CHECK-NEXT:   %9 = call i32 @other(i64 %8, i32 %0)
// CHECK-NEXT:   %10 = call i32 @other(i64 %8, i32 %9)
// CHECK-NEXT:   %11 = call i32 @other(i64 %8, i32 %1)
// CHECK-NEXT:   %12 = add i64 %5, 1
// CHECK-NEXT:   br label %[[ARGS_bb2]]
^bb3:   // pred: ^bb2
  %5 = llvm.call @body_args(%3) : (!llvm.i64) -> !llvm.i64
  %6 = llvm.call @other(%5, %arg0) : (!llvm.i64, !llvm.i32) -> !llvm.i32
  %7 = llvm.call @other(%5, %6) : (!llvm.i64, !llvm.i32) -> !llvm.i32
  %8 = llvm.call @other(%5, %arg1) : (!llvm.i64, !llvm.i32) -> !llvm.i32
  %9 = llvm.mlir.constant(1 : index) : !llvm.i64
  %10 = llvm.add %3, %9 : !llvm.i64
  llvm.br ^bb2(%10 : !llvm.i64)

// CHECK: [[ARGS_bb4]]:
// CHECK-NEXT:   %14 = call i32 @other(i64 0, i32 0)
// CHECK-NEXT:   ret i32 %14
^bb4:   // pred: ^bb2
  %11 = llvm.mlir.constant(0 : index) : !llvm.i64
  %12 = llvm.call @other(%11, %0) : (!llvm.i64, !llvm.i32) -> !llvm.i32
  llvm.return %12 : !llvm.i32
}

// CHECK: declare void @pre(i64)
llvm.func @pre(!llvm.i64)

// CHECK: declare void @body2(i64, i64)
llvm.func @body2(!llvm.i64, !llvm.i64)

// CHECK: declare void @post(i64)
llvm.func @post(!llvm.i64)

// CHECK-LABEL: define void @imperfectly_nested_loops()
// CHECK-NEXT:   br label %[[IMPER_bb1:[0-9]+]]
llvm.func @imperfectly_nested_loops() {
  llvm.br ^bb1

// CHECK: [[IMPER_bb1]]:
// CHECK-NEXT:   br label %[[IMPER_bb2:[0-9]+]]
^bb1:   // pred: ^bb0
  %0 = llvm.mlir.constant(0 : index) : !llvm.i64
  %1 = llvm.mlir.constant(42 : index) : !llvm.i64
  llvm.br ^bb2(%0 : !llvm.i64)

// CHECK: [[IMPER_bb2]]:
// CHECK-NEXT:   %3 = phi i64 [ %13, %[[IMPER_bb7:[0-9]+]] ], [ 0, %[[IMPER_bb1]] ]
// CHECK-NEXT:   %4 = icmp slt i64 %3, 42
// CHECK-NEXT:   br i1 %4, label %[[IMPER_bb3:[0-9]+]], label %[[IMPER_bb8:[0-9]+]]
^bb2(%2: !llvm.i64): // 2 preds: ^bb1, ^bb7
  %3 = llvm.icmp "slt" %2, %1 : !llvm.i64
  llvm.cond_br %3, ^bb3, ^bb8

// CHECK: [[IMPER_bb3]]:
// CHECK-NEXT:   call void @pre(i64 %3)
// CHECK-NEXT:   br label %[[IMPER_bb4:[0-9]+]]
^bb3:   // pred: ^bb2
  llvm.call @pre(%2) : (!llvm.i64) -> ()
  llvm.br ^bb4

// CHECK: [[IMPER_bb4]]:
// CHECK-NEXT:   br label %[[IMPER_bb5:[0-9]+]]
^bb4:   // pred: ^bb3
  %4 = llvm.mlir.constant(7 : index) : !llvm.i64
  %5 = llvm.mlir.constant(56 : index) : !llvm.i64
  llvm.br ^bb5(%4 : !llvm.i64)

// CHECK: [[IMPER_bb5]]:
// CHECK-NEXT:   %8 = phi i64 [ %11, %[[IMPER_bb6:[0-9]+]] ], [ 7, %[[IMPER_bb4]] ]
// CHECK-NEXT:   %9 = icmp slt i64 %8, 56
// CHECK-NEXT:   br i1 %9, label %[[IMPER_bb6]], label %[[IMPER_bb7]]
^bb5(%6: !llvm.i64): // 2 preds: ^bb4, ^bb6
  %7 = llvm.icmp "slt" %6, %5 : !llvm.i64
  llvm.cond_br %7, ^bb6, ^bb7

// CHECK: [[IMPER_bb6]]:
// CHECK-NEXT:   call void @body2(i64 %3, i64 %8)
// CHECK-NEXT:   %11 = add i64 %8, 2
// CHECK-NEXT:   br label %[[IMPER_bb5]]
^bb6:   // pred: ^bb5
  llvm.call @body2(%2, %6) : (!llvm.i64, !llvm.i64) -> ()
  %8 = llvm.mlir.constant(2 : index) : !llvm.i64
  %9 = llvm.add %6, %8 : !llvm.i64
  llvm.br ^bb5(%9 : !llvm.i64)

// CHECK: [[IMPER_bb7]]:
// CHECK-NEXT:   call void @post(i64 %3)
// CHECK-NEXT:   %13 = add i64 %3, 1
// CHECK-NEXT:   br label %[[IMPER_bb2]]
^bb7:   // pred: ^bb5
  llvm.call @post(%2) : (!llvm.i64) -> ()
  %10 = llvm.mlir.constant(1 : index) : !llvm.i64
  %11 = llvm.add %2, %10 : !llvm.i64
  llvm.br ^bb2(%11 : !llvm.i64)

// CHECK: [[IMPER_bb8]]:
// CHECK-NEXT:   ret void
^bb8:   // pred: ^bb2
  llvm.return
}

// CHECK: declare void @mid(i64)
llvm.func @mid(!llvm.i64)

// CHECK: declare void @body3(i64, i64)
llvm.func @body3(!llvm.i64, !llvm.i64)

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
  %0 = llvm.mlir.constant(0 : index) : !llvm.i64
  %1 = llvm.mlir.constant(42 : index) : !llvm.i64
  llvm.br ^bb2(%0 : !llvm.i64)
^bb2(%2: !llvm.i64):	// 2 preds: ^bb1, ^bb11
  %3 = llvm.icmp "slt" %2, %1 : !llvm.i64
  llvm.cond_br %3, ^bb3, ^bb12
^bb3:	// pred: ^bb2
  llvm.call @pre(%2) : (!llvm.i64) -> ()
  llvm.br ^bb4
^bb4:	// pred: ^bb3
  %4 = llvm.mlir.constant(7 : index) : !llvm.i64
  %5 = llvm.mlir.constant(56 : index) : !llvm.i64
  llvm.br ^bb5(%4 : !llvm.i64)
^bb5(%6: !llvm.i64):	// 2 preds: ^bb4, ^bb6
  %7 = llvm.icmp "slt" %6, %5 : !llvm.i64
  llvm.cond_br %7, ^bb6, ^bb7
^bb6:	// pred: ^bb5
  llvm.call @body2(%2, %6) : (!llvm.i64, !llvm.i64) -> ()
  %8 = llvm.mlir.constant(2 : index) : !llvm.i64
  %9 = llvm.add %6, %8 : !llvm.i64
  llvm.br ^bb5(%9 : !llvm.i64)
^bb7:	// pred: ^bb5
  llvm.call @mid(%2) : (!llvm.i64) -> ()
  llvm.br ^bb8
^bb8:	// pred: ^bb7
  %10 = llvm.mlir.constant(18 : index) : !llvm.i64
  %11 = llvm.mlir.constant(37 : index) : !llvm.i64
  llvm.br ^bb9(%10 : !llvm.i64)
^bb9(%12: !llvm.i64):	// 2 preds: ^bb8, ^bb10
  %13 = llvm.icmp "slt" %12, %11 : !llvm.i64
  llvm.cond_br %13, ^bb10, ^bb11
^bb10:	// pred: ^bb9
  llvm.call @body3(%2, %12) : (!llvm.i64, !llvm.i64) -> ()
  %14 = llvm.mlir.constant(3 : index) : !llvm.i64
  %15 = llvm.add %12, %14 : !llvm.i64
  llvm.br ^bb9(%15 : !llvm.i64)
^bb11:	// pred: ^bb9
  llvm.call @post(%2) : (!llvm.i64) -> ()
  %16 = llvm.mlir.constant(1 : index) : !llvm.i64
  %17 = llvm.add %2, %16 : !llvm.i64
  llvm.br ^bb2(%17 : !llvm.i64)
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
// MemRef type conversion, allocation and communication with functions.
//

// CHECK-LABEL: define void @memref_alloc()
llvm.func @memref_alloc() {
// CHECK-NEXT: %{{[0-9]+}} = call i8* @malloc(i64 400)
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float* } undef, float* %{{[0-9]+}}, 0
  %0 = llvm.mlir.constant(10 : index) : !llvm.i64
  %1 = llvm.mlir.constant(10 : index) : !llvm.i64
  %2 = llvm.mul %0, %1 : !llvm.i64
  %3 = llvm.mlir.undef : !llvm.struct<(ptr<float>)>
  %4 = llvm.mlir.constant(4 : index) : !llvm.i64
  %5 = llvm.mul %2, %4 : !llvm.i64
  %6 = llvm.call @malloc(%5) : (!llvm.i64) -> !llvm.ptr<i8>
  %7 = llvm.bitcast %6 : !llvm.ptr<i8> to !llvm.ptr<float>
  %8 = llvm.insertvalue %7, %3[0] : !llvm.struct<(ptr<float>)>
// CHECK-NEXT: ret void
  llvm.return
}

// CHECK-LABEL: declare i64 @get_index()
llvm.func @get_index() -> !llvm.i64

// CHECK-LABEL: define void @store_load_static()
llvm.func @store_load_static() {
^bb0:
// CHECK-NEXT: %{{[0-9]+}} = call i8* @malloc(i64 40)
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float* } undef, float* %{{[0-9]+}}, 0
  %0 = llvm.mlir.constant(10 : index) : !llvm.i64
  %1 = llvm.mlir.undef : !llvm.struct<(ptr<float>)>
  %2 = llvm.mlir.constant(4 : index) : !llvm.i64
  %3 = llvm.mul %0, %2 : !llvm.i64
  %4 = llvm.call @malloc(%3) : (!llvm.i64) -> !llvm.ptr<i8>
  %5 = llvm.bitcast %4 : !llvm.ptr<i8> to !llvm.ptr<float>
  %6 = llvm.insertvalue %5, %1[0] : !llvm.struct<(ptr<float>)>
  %7 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
  llvm.br ^bb1
^bb1:   // pred: ^bb0
  %8 = llvm.mlir.constant(0 : index) : !llvm.i64
  %9 = llvm.mlir.constant(10 : index) : !llvm.i64
  llvm.br ^bb2(%8 : !llvm.i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb2(%10: !llvm.i64):        // 2 preds: ^bb1, ^bb3
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 10
  %11 = llvm.icmp "slt" %10, %9 : !llvm.i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %11, ^bb3, ^bb4
^bb3:   // pred: ^bb2
// CHECK: %{{[0-9]+}} = extractvalue { float* } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 1.000000e+00, float* %{{[0-9]+}}
  %12 = llvm.mlir.constant(10 : index) : !llvm.i64
  %13 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<float>)>
  %14 = llvm.getelementptr %13[%10] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  llvm.store %7, %14 : !llvm.ptr<float>
  %15 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %16 = llvm.add %10, %15 : !llvm.i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb2(%16 : !llvm.i64)
^bb4:   // pred: ^bb2
  llvm.br ^bb5
^bb5:   // pred: ^bb4
  %17 = llvm.mlir.constant(0 : index) : !llvm.i64
  %18 = llvm.mlir.constant(10 : index) : !llvm.i64
  llvm.br ^bb6(%17 : !llvm.i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb6(%19: !llvm.i64):        // 2 preds: ^bb5, ^bb7
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, 10
  %20 = llvm.icmp "slt" %19, %18 : !llvm.i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %20, ^bb7, ^bb8
^bb7:   // pred: ^bb6
// CHECK:      %{{[0-9]+}} = extractvalue { float* } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, float* %{{[0-9]+}}
  %21 = llvm.mlir.constant(10 : index) : !llvm.i64
  %22 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<float>)>
  %23 = llvm.getelementptr %22[%19] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  %24 = llvm.load %23 : !llvm.ptr<float>
  %25 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %26 = llvm.add %19, %25 : !llvm.i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb6(%26 : !llvm.i64)
^bb8:   // pred: ^bb6
// CHECK: ret void
  llvm.return
}

// CHECK-LABEL: define void @store_load_dynamic(i64 {{%.*}})
llvm.func @store_load_dynamic(%arg0: !llvm.i64) {
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call i8* @malloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } undef, float* %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
  %0 = llvm.mlir.undef : !llvm.struct<(ptr<float>, i64)>
  %1 = llvm.mlir.constant(4 : index) : !llvm.i64
  %2 = llvm.mul %arg0, %1 : !llvm.i64
  %3 = llvm.call @malloc(%2) : (!llvm.i64) -> !llvm.ptr<i8>
  %4 = llvm.bitcast %3 : !llvm.ptr<i8> to !llvm.ptr<float>
  %5 = llvm.insertvalue %4, %0[0] : !llvm.struct<(ptr<float>, i64)>
  %6 = llvm.insertvalue %arg0, %5[1] : !llvm.struct<(ptr<float>, i64)>
  %7 = llvm.mlir.constant(1.000000e+00 : f32) : !llvm.float
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb1
^bb1:   // pred: ^bb0
  %8 = llvm.mlir.constant(0 : index) : !llvm.i64
  llvm.br ^bb2(%8 : !llvm.i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb2(%9: !llvm.i64): // 2 preds: ^bb1, ^bb3
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, %{{[0-9]+}}
  %10 = llvm.icmp "slt" %9, %arg0 : !llvm.i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %10, ^bb3, ^bb4
^bb3:   // pred: ^bb2
// CHECK:      %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 1.000000e+00, float* %{{[0-9]+}}
  %11 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<float>, i64)>
  %12 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<float>, i64)>
  %13 = llvm.getelementptr %12[%9] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  llvm.store %7, %13 : !llvm.ptr<float>
  %14 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %15 = llvm.add %9, %14 : !llvm.i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb2(%15 : !llvm.i64)
^bb4:   // pred: ^bb3
  llvm.br ^bb5
^bb5:   // pred: ^bb4
  %16 = llvm.mlir.constant(0 : index) : !llvm.i64
  llvm.br ^bb6(%16 : !llvm.i64)
// CHECK: %{{[0-9]+}} = phi i64 [ %{{[0-9]+}}, %{{[0-9]+}} ], [ 0, %{{[0-9]+}} ]
^bb6(%17: !llvm.i64):        // 2 preds: ^bb5, ^bb7
// CHECK-NEXT: %{{[0-9]+}} = icmp slt i64 %{{[0-9]+}}, %{{[0-9]+}}
  %18 = llvm.icmp "slt" %17, %arg0 : !llvm.i64
// CHECK-NEXT: br i1 %{{[0-9]+}}, label %{{[0-9]+}}, label %{{[0-9]+}}
  llvm.cond_br %18, ^bb7, ^bb8
^bb7:   // pred: ^bb6
// CHECK:      %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, float* %{{[0-9]+}}
  %19 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<float>, i64)>
  %20 = llvm.extractvalue %6[0] : !llvm.struct<(ptr<float>, i64)>
  %21 = llvm.getelementptr %20[%17] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  %22 = llvm.load %21 : !llvm.ptr<float>
  %23 = llvm.mlir.constant(1 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
  %24 = llvm.add %17, %23 : !llvm.i64
// CHECK-NEXT: br label %{{[0-9]+}}
  llvm.br ^bb6(%24 : !llvm.i64)
^bb8:   // pred: ^bb6
// CHECK: ret void
  llvm.return
}

// CHECK-LABEL: define void @store_load_mixed(i64 {{%.*}})
llvm.func @store_load_mixed(%arg0: !llvm.i64) {
  %0 = llvm.mlir.constant(10 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = mul i64 2, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 10
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call i8* @malloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64, i64 } undef, float* %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64, i64 } %{{[0-9]+}}, i64 10, 2
  %1 = llvm.mlir.constant(2 : index) : !llvm.i64
  %2 = llvm.mlir.constant(4 : index) : !llvm.i64
  %3 = llvm.mul %1, %arg0 : !llvm.i64
  %4 = llvm.mul %3, %2 : !llvm.i64
  %5 = llvm.mul %4, %0 : !llvm.i64
  %6 = llvm.mlir.undef : !llvm.struct<(ptr<float>, i64, i64)>
  %7 = llvm.mlir.constant(4 : index) : !llvm.i64
  %8 = llvm.mul %5, %7 : !llvm.i64
  %9 = llvm.call @malloc(%8) : (!llvm.i64) -> !llvm.ptr<i8>
  %10 = llvm.bitcast %9 : !llvm.ptr<i8> to !llvm.ptr<float>
  %11 = llvm.insertvalue %10, %6[0] : !llvm.struct<(ptr<float>, i64, i64)>
  %12 = llvm.insertvalue %arg0, %11[1] : !llvm.struct<(ptr<float>, i64, i64)>
  %13 = llvm.insertvalue %0, %12[2] : !llvm.struct<(ptr<float>, i64, i64)>

// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
  %14 = llvm.mlir.constant(1 : index) : !llvm.i64
  %15 = llvm.mlir.constant(2 : index) : !llvm.i64
  %16 = llvm.call @get_index() : () -> !llvm.i64
  %17 = llvm.call @get_index() : () -> !llvm.i64
  %18 = llvm.mlir.constant(4.200000e+01 : f32) : !llvm.float
  %19 = llvm.mlir.constant(2 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 1, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  %20 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<float>, i64, i64)>
  %21 = llvm.mlir.constant(4 : index) : !llvm.i64
  %22 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<float>, i64, i64)>
  %23 = llvm.mul %14, %20 : !llvm.i64
  %24 = llvm.add %23, %15 : !llvm.i64
  %25 = llvm.mul %24, %21 : !llvm.i64
  %26 = llvm.add %25, %16 : !llvm.i64
  %27 = llvm.mul %26, %22 : !llvm.i64
  %28 = llvm.add %27, %17 : !llvm.i64
  %29 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<float>, i64, i64)>
  %30 = llvm.getelementptr %29[%28] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  llvm.store %18, %30 : !llvm.ptr<float>
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 2
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = load float, float* %{{[0-9]+}}
  %31 = llvm.mlir.constant(2 : index) : !llvm.i64
  %32 = llvm.extractvalue %13[1] : !llvm.struct<(ptr<float>, i64, i64)>
  %33 = llvm.mlir.constant(4 : index) : !llvm.i64
  %34 = llvm.extractvalue %13[2] : !llvm.struct<(ptr<float>, i64, i64)>
  %35 = llvm.mul %17, %32 : !llvm.i64
  %36 = llvm.add %35, %16 : !llvm.i64
  %37 = llvm.mul %36, %33 : !llvm.i64
  %38 = llvm.add %37, %15 : !llvm.i64
  %39 = llvm.mul %38, %34 : !llvm.i64
  %40 = llvm.add %39, %14 : !llvm.i64
  %41 = llvm.extractvalue %13[0] : !llvm.struct<(ptr<float>, i64, i64)>
  %42 = llvm.getelementptr %41[%40] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  %43 = llvm.load %42 : !llvm.ptr<float>
// CHECK-NEXT: ret void
  llvm.return
}

// CHECK-LABEL: define { float*, i64 } @memref_args_rets({ float* } {{%.*}}, { float*, i64 } {{%.*}}, { float*, i64 } {{%.*}})
llvm.func @memref_args_rets(%arg0: !llvm.struct<(ptr<float>)>, %arg1: !llvm.struct<(ptr<float>, i64)>, %arg2: !llvm.struct<(ptr<float>, i64)>) -> !llvm.struct<(ptr<float>, i64)> {
  %0 = llvm.mlir.constant(7 : index) : !llvm.i64
// CHECK-NEXT: %{{[0-9]+}} = call i64 @get_index()
  %1 = llvm.call @get_index() : () -> !llvm.i64
  %2 = llvm.mlir.constant(4.200000e+01 : f32) : !llvm.float
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float* } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 7
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  %3 = llvm.mlir.constant(10 : index) : !llvm.i64
  %4 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<float>)>
  %5 = llvm.getelementptr %4[%0] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  llvm.store %2, %5 : !llvm.ptr<float>
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 7
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  %6 = llvm.extractvalue %arg1[1] : !llvm.struct<(ptr<float>, i64)>
  %7 = llvm.extractvalue %arg1[0] : !llvm.struct<(ptr<float>, i64)>
  %8 = llvm.getelementptr %7[%0] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  llvm.store %2, %8 : !llvm.ptr<float>
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 1
// CHECK-NEXT: %{{[0-9]+}} = mul i64 7, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = add i64 %{{[0-9]+}}, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = extractvalue { float*, i64 } %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = getelementptr float, float* %{{[0-9]+}}, i64 %{{[0-9]+}}
// CHECK-NEXT: store float 4.200000e+01, float* %{{[0-9]+}}
  %9 = llvm.mlir.constant(10 : index) : !llvm.i64
  %10 = llvm.extractvalue %arg2[1] : !llvm.struct<(ptr<float>, i64)>
  %11 = llvm.mul %0, %10 : !llvm.i64
  %12 = llvm.add %11, %1 : !llvm.i64
  %13 = llvm.extractvalue %arg2[0] : !llvm.struct<(ptr<float>, i64)>
  %14 = llvm.getelementptr %13[%12] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  llvm.store %2, %14 : !llvm.ptr<float>
// CHECK-NEXT: %{{[0-9]+}} = mul i64 10, %{{[0-9]+}}
// CHECK-NEXT: %{{[0-9]+}} = mul i64 %{{[0-9]+}}, 4
// CHECK-NEXT: %{{[0-9]+}} = call i8* @malloc(i64 %{{[0-9]+}})
// CHECK-NEXT: %{{[0-9]+}} = bitcast i8* %{{[0-9]+}} to float*
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } undef, float* %{{[0-9]+}}, 0
// CHECK-NEXT: %{{[0-9]+}} = insertvalue { float*, i64 } %{{[0-9]+}}, i64 %{{[0-9]+}}, 1
  %15 = llvm.mlir.constant(10 : index) : !llvm.i64
  %16 = llvm.mul %15, %1 : !llvm.i64
  %17 = llvm.mlir.undef : !llvm.struct<(ptr<float>, i64)>
  %18 = llvm.mlir.constant(4 : index) : !llvm.i64
  %19 = llvm.mul %16, %18 : !llvm.i64
  %20 = llvm.call @malloc(%19) : (!llvm.i64) -> !llvm.ptr<i8>
  %21 = llvm.bitcast %20 : !llvm.ptr<i8> to !llvm.ptr<float>
  %22 = llvm.insertvalue %21, %17[0] : !llvm.struct<(ptr<float>, i64)>
  %23 = llvm.insertvalue %1, %22[1] : !llvm.struct<(ptr<float>, i64)>
// CHECK-NEXT: ret { float*, i64 } %{{[0-9]+}}
  llvm.return %23 : !llvm.struct<(ptr<float>, i64)>
}


// CHECK-LABEL: define i64 @memref_dim({ float*, i64, i64 } {{%.*}})
llvm.func @memref_dim(%arg0: !llvm.struct<(ptr<float>, i64, i64)>) -> !llvm.i64 {
// Expecting this to create an LLVM constant.
  %0 = llvm.mlir.constant(42 : index) : !llvm.i64
// CHECK-NEXT: %2 = extractvalue { float*, i64, i64 } %0, 1
  %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<float>, i64, i64)>
// Expecting this to create an LLVM constant.
  %2 = llvm.mlir.constant(10 : index) : !llvm.i64
// CHECK-NEXT: %3 = extractvalue { float*, i64, i64 } %0, 2
  %3 = llvm.extractvalue %arg0[2] : !llvm.struct<(ptr<float>, i64, i64)>
// Checking that the constant for d0 has been created.
// CHECK-NEXT: %4 = add i64 42, %2
  %4 = llvm.add %0, %1 : !llvm.i64
// Checking that the constant for d2 has been created.
// CHECK-NEXT: %5 = add i64 10, %3
  %5 = llvm.add %2, %3 : !llvm.i64
// CHECK-NEXT: %6 = add i64 %4, %5
  %6 = llvm.add %4, %5 : !llvm.i64
// CHECK-NEXT: ret i64 %6
  llvm.return %6 : !llvm.i64
}

llvm.func @get_i64() -> !llvm.i64
llvm.func @get_f32() -> !llvm.float
llvm.func @get_memref() -> !llvm.struct<(ptr<float>, i64, i64)>

// CHECK-LABEL: define { i64, float, { float*, i64, i64 } } @multireturn()
llvm.func @multireturn() -> !llvm.struct<(i64, float, struct<(ptr<float>, i64, i64)>)> {
  %0 = llvm.call @get_i64() : () -> !llvm.i64
  %1 = llvm.call @get_f32() : () -> !llvm.float
  %2 = llvm.call @get_memref() : () -> !llvm.struct<(ptr<float>, i64, i64)>
// CHECK:        %{{[0-9]+}} = insertvalue { i64, float, { float*, i64, i64 } } undef, i64 %{{[0-9]+}}, 0
// CHECK-NEXT:   %{{[0-9]+}} = insertvalue { i64, float, { float*, i64, i64 } } %{{[0-9]+}}, float %{{[0-9]+}}, 1
// CHECK-NEXT:   %{{[0-9]+}} = insertvalue { i64, float, { float*, i64, i64 } } %{{[0-9]+}}, { float*, i64, i64 } %{{[0-9]+}}, 2
// CHECK-NEXT:   ret { i64, float, { float*, i64, i64 } } %{{[0-9]+}}
  %3 = llvm.mlir.undef : !llvm.struct<(i64, float, struct<(ptr<float>, i64, i64)>)>
  %4 = llvm.insertvalue %0, %3[0] : !llvm.struct<(i64, float, struct<(ptr<float>, i64, i64)>)>
  %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(i64, float, struct<(ptr<float>, i64, i64)>)>
  %6 = llvm.insertvalue %2, %5[2] : !llvm.struct<(i64, float, struct<(ptr<float>, i64, i64)>)>
  llvm.return %6 : !llvm.struct<(i64, float, struct<(ptr<float>, i64, i64)>)>
}


// CHECK-LABEL: define void @multireturn_caller()
llvm.func @multireturn_caller() {
// CHECK-NEXT:   %1 = call { i64, float, { float*, i64, i64 } } @multireturn()
// CHECK-NEXT:   [[ret0:%[0-9]+]] = extractvalue { i64, float, { float*, i64, i64 } } %1, 0
// CHECK-NEXT:   [[ret1:%[0-9]+]] = extractvalue { i64, float, { float*, i64, i64 } } %1, 1
// CHECK-NEXT:   [[ret2:%[0-9]+]] = extractvalue { i64, float, { float*, i64, i64 } } %1, 2
  %0 = llvm.call @multireturn() : () -> !llvm.struct<(i64, float, struct<(ptr<float>, i64, i64)>)>
  %1 = llvm.extractvalue %0[0] : !llvm.struct<(i64, float, struct<(ptr<float>, i64, i64)>)>
  %2 = llvm.extractvalue %0[1] : !llvm.struct<(i64, float, struct<(ptr<float>, i64, i64)>)>
  %3 = llvm.extractvalue %0[2] : !llvm.struct<(i64, float, struct<(ptr<float>, i64, i64)>)>
  %4 = llvm.mlir.constant(42) : !llvm.i64
// CHECK:   add i64 [[ret0]], 42
  %5 = llvm.add %1, %4 : !llvm.i64
  %6 = llvm.mlir.constant(4.200000e+01 : f32) : !llvm.float
// CHECK:   fadd float [[ret1]], 4.200000e+01
  %7 = llvm.fadd %2, %6 : !llvm.float
  %8 = llvm.mlir.constant(0 : index) : !llvm.i64
  %9 = llvm.mlir.constant(42 : index) : !llvm.i64
// CHECK:   extractvalue { float*, i64, i64 } [[ret2]], 0
  %10 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<float>, i64, i64)>
  %11 = llvm.mlir.constant(10 : index) : !llvm.i64
  %12 = llvm.extractvalue %3[2] : !llvm.struct<(ptr<float>, i64, i64)>
  %13 = llvm.mul %8, %10 : !llvm.i64
  %14 = llvm.add %13, %8 : !llvm.i64
  %15 = llvm.mul %14, %11 : !llvm.i64
  %16 = llvm.add %15, %8 : !llvm.i64
  %17 = llvm.mul %16, %12 : !llvm.i64
  %18 = llvm.add %17, %8 : !llvm.i64
  %19 = llvm.extractvalue %3[0] : !llvm.struct<(ptr<float>, i64, i64)>
  %20 = llvm.getelementptr %19[%18] : (!llvm.ptr<float>, !llvm.i64) -> !llvm.ptr<float>
  %21 = llvm.load %20 : !llvm.ptr<float>
  llvm.return
}

// CHECK-LABEL: define <4 x float> @vector_ops(<4 x float> {{%.*}}, <4 x i1> {{%.*}}, <4 x i64> {{%.*}})
llvm.func @vector_ops(%arg0: !llvm.vec<4 x float>, %arg1: !llvm.vec<4 x i1>, %arg2: !llvm.vec<4 x i64>) -> !llvm.vec<4 x float> {
  %0 = llvm.mlir.constant(dense<4.200000e+01> : vector<4xf32>) : !llvm.vec<4 x float>
// CHECK-NEXT: %4 = fadd <4 x float> %0, <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
  %1 = llvm.fadd %arg0, %0 : !llvm.vec<4 x float>
// CHECK-NEXT: %5 = select <4 x i1> %1, <4 x float> %4, <4 x float> %0
  %2 = llvm.select %arg1, %1, %arg0 : !llvm.vec<4 x i1>, !llvm.vec<4 x float>
// CHECK-NEXT: %6 = sdiv <4 x i64> %2, %2
  %3 = llvm.sdiv %arg2, %arg2 : !llvm.vec<4 x i64>
// CHECK-NEXT: %7 = udiv <4 x i64> %2, %2
  %4 = llvm.udiv %arg2, %arg2 : !llvm.vec<4 x i64>
// CHECK-NEXT: %8 = srem <4 x i64> %2, %2
  %5 = llvm.srem %arg2, %arg2 : !llvm.vec<4 x i64>
// CHECK-NEXT: %9 = urem <4 x i64> %2, %2
  %6 = llvm.urem %arg2, %arg2 : !llvm.vec<4 x i64>
// CHECK-NEXT: %10 = fdiv <4 x float> %0, <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
  %7 = llvm.fdiv %arg0, %0 : !llvm.vec<4 x float>
// CHECK-NEXT: %11 = frem <4 x float> %0, <float 4.200000e+01, float 4.200000e+01, float 4.200000e+01, float 4.200000e+01>
  %8 = llvm.frem %arg0, %0 : !llvm.vec<4 x float>
// CHECK-NEXT: %12 = and <4 x i64> %2, %2
  %9 = llvm.and %arg2, %arg2 : !llvm.vec<4 x i64>
// CHECK-NEXT: %13 = or <4 x i64> %2, %2
  %10 = llvm.or %arg2, %arg2 : !llvm.vec<4 x i64>
// CHECK-NEXT: %14 = xor <4 x i64> %2, %2
  %11 = llvm.xor %arg2, %arg2 : !llvm.vec<4 x i64>
// CHECK-NEXT: %15 = shl <4 x i64> %2, %2
  %12 = llvm.shl %arg2, %arg2 : !llvm.vec<4 x i64>
// CHECK-NEXT: %16 = lshr <4 x i64> %2, %2
  %13 = llvm.lshr %arg2, %arg2 : !llvm.vec<4 x i64>
// CHECK-NEXT: %17 = ashr <4 x i64> %2, %2
  %14 = llvm.ashr %arg2, %arg2 : !llvm.vec<4 x i64>
// CHECK-NEXT:    ret <4 x float> %4
  llvm.return %1 : !llvm.vec<4 x float>
}

// CHECK-LABEL: @vector_splat_1d
llvm.func @vector_splat_1d() -> !llvm.vec<4 x float> {
  // CHECK: ret <4 x float> zeroinitializer
  %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<4xf32>) : !llvm.vec<4 x float>
  llvm.return %0 : !llvm.vec<4 x float>
}

// CHECK-LABEL: @vector_splat_2d
llvm.func @vector_splat_2d() -> !llvm.array<4 x vec<16 x float>> {
  // CHECK: ret [4 x <16 x float>] zeroinitializer
  %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<4x16xf32>) : !llvm.array<4 x vec<16 x float>>
  llvm.return %0 : !llvm.array<4 x vec<16 x float>>
}

// CHECK-LABEL: @vector_splat_3d
llvm.func @vector_splat_3d() -> !llvm.array<4 x array<16 x vec<4 x float>>> {
  // CHECK: ret [4 x [16 x <4 x float>]] zeroinitializer
  %0 = llvm.mlir.constant(dense<0.000000e+00> : vector<4x16x4xf32>) : !llvm.array<4 x array<16 x vec<4 x float>>>
  llvm.return %0 : !llvm.array<4 x array<16 x vec<4 x float>>>
}

// CHECK-LABEL: @vector_splat_nonzero
llvm.func @vector_splat_nonzero() -> !llvm.vec<4 x float> {
  // CHECK: ret <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %0 = llvm.mlir.constant(dense<1.000000e+00> : vector<4xf32>) : !llvm.vec<4 x float>
  llvm.return %0 : !llvm.vec<4 x float>
}

// CHECK-LABEL: @ops
llvm.func @ops(%arg0: !llvm.float, %arg1: !llvm.float, %arg2: !llvm.i32, %arg3: !llvm.i32) -> !llvm.struct<(float, i32)> {
// CHECK-NEXT: fsub float %0, %1
  %0 = llvm.fsub %arg0, %arg1 : !llvm.float
// CHECK-NEXT: %6 = sub i32 %2, %3
  %1 = llvm.sub %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %7 = icmp slt i32 %2, %6
  %2 = llvm.icmp "slt" %arg2, %1 : !llvm.i32
// CHECK-NEXT: %8 = select i1 %7, i32 %2, i32 %6
  %3 = llvm.select %2, %arg2, %1 : !llvm.i1, !llvm.i32
// CHECK-NEXT: %9 = sdiv i32 %2, %3
  %4 = llvm.sdiv %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %10 = udiv i32 %2, %3
  %5 = llvm.udiv %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %11 = srem i32 %2, %3
  %6 = llvm.srem %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %12 = urem i32 %2, %3
  %7 = llvm.urem %arg2, %arg3 : !llvm.i32

  %8 = llvm.mlir.undef : !llvm.struct<(float, i32)>
  %9 = llvm.insertvalue %0, %8[0] : !llvm.struct<(float, i32)>
  %10 = llvm.insertvalue %3, %9[1] : !llvm.struct<(float, i32)>

// CHECK: %15 = fdiv float %0, %1
  %11 = llvm.fdiv %arg0, %arg1 : !llvm.float
// CHECK-NEXT: %16 = frem float %0, %1
  %12 = llvm.frem %arg0, %arg1 : !llvm.float

// CHECK-NEXT: %17 = and i32 %2, %3
  %13 = llvm.and %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %18 = or i32 %2, %3
  %14 = llvm.or %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %19 = xor i32 %2, %3
  %15 = llvm.xor %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %20 = shl i32 %2, %3
  %16 = llvm.shl %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %21 = lshr i32 %2, %3
  %17 = llvm.lshr %arg2, %arg3 : !llvm.i32
// CHECK-NEXT: %22 = ashr i32 %2, %3
  %18 = llvm.ashr %arg2, %arg3 : !llvm.i32

// CHECK-NEXT: fneg float %0
  %19 = llvm.fneg %arg0 : !llvm.float

  llvm.return %10 : !llvm.struct<(float, i32)>
}

//
// Indirect function calls
//

// CHECK-LABEL: define void @indirect_const_call(i64 {{%.*}})
llvm.func @indirect_const_call(%arg0: !llvm.i64) {
// CHECK-NEXT:  call void @body(i64 %0)
  %0 = llvm.mlir.addressof @body : !llvm.ptr<func<void (i64)>>
  llvm.call %0(%arg0) : (!llvm.i64) -> ()
// CHECK-NEXT:  ret void
  llvm.return
}

// CHECK-LABEL: define i32 @indirect_call(i32 (float)* {{%.*}}, float {{%.*}})
llvm.func @indirect_call(%arg0: !llvm.ptr<func<i32 (float)>>, %arg1: !llvm.float) -> !llvm.i32 {
// CHECK-NEXT:  %3 = call i32 %0(float %1)
  %0 = llvm.call %arg0(%arg1) : (!llvm.float) -> !llvm.i32
// CHECK-NEXT:  ret i32 %3
  llvm.return %0 : !llvm.i32
}

//
// Check that we properly construct phi nodes in the blocks that have the same
// predecessor more than once.
//

// CHECK-LABEL: define void @cond_br_arguments(i1 {{%.*}}, i1 {{%.*}})
llvm.func @cond_br_arguments(%arg0: !llvm.i1, %arg1: !llvm.i1) {
// CHECK-NEXT:   br i1 %0, label %3, label %5
  llvm.cond_br %arg0, ^bb1(%arg0 : !llvm.i1), ^bb2

// CHECK:      3:
// CHECK-NEXT:   %4 = phi i1 [ %1, %5 ], [ %0, %2 ]
^bb1(%0 : !llvm.i1):
// CHECK-NEXT:   ret void
  llvm.return

// CHECK:      5:
^bb2:
// CHECK-NEXT:   br label %3
  llvm.br ^bb1(%arg1 : !llvm.i1)
}

// CHECK-LABEL: define void @llvm_noalias(float* noalias {{%*.}})
llvm.func @llvm_noalias(%arg0: !llvm.ptr<float> {llvm.noalias = true}) {
  llvm.return
}

// CHECK-LABEL: define void @llvm_align(float* align 4 {{%*.}})
llvm.func @llvm_align(%arg0: !llvm.ptr<float> {llvm.align = 4}) {
  llvm.return
}

// CHECK-LABEL: @llvm_varargs(...)
llvm.func @llvm_varargs(...)

llvm.func @intpointerconversion(%arg0 : !llvm.i32) -> !llvm.i32 {
// CHECK:      %2 = inttoptr i32 %0 to i32*
// CHECK-NEXT: %3 = ptrtoint i32* %2 to i32
  %1 = llvm.inttoptr %arg0 : !llvm.i32 to !llvm.ptr<i32>
  %2 = llvm.ptrtoint %1 : !llvm.ptr<i32> to !llvm.i32
  llvm.return %2 : !llvm.i32
}

llvm.func @fpconversion(%arg0 : !llvm.i32) -> !llvm.i32 {
// CHECK:      %2 = sitofp i32 %0 to float
// CHECK-NEXT: %3 = fptosi float %2 to i32
// CHECK-NEXT: %4 = uitofp i32 %3 to float
// CHECK-NEXT: %5 = fptoui float %4 to i32
  %1 = llvm.sitofp %arg0 : !llvm.i32 to !llvm.float
  %2 = llvm.fptosi %1 : !llvm.float to !llvm.i32
  %3 = llvm.uitofp %2 : !llvm.i32 to !llvm.float
  %4 = llvm.fptoui %3 : !llvm.float to !llvm.i32
  llvm.return %4 : !llvm.i32
}

// CHECK-LABEL: @addrspace
llvm.func @addrspace(%arg0 : !llvm.ptr<i32>) -> !llvm.ptr<i32, 2> {
// CHECK: %2 = addrspacecast i32* %0 to i32 addrspace(2)*
  %1 = llvm.addrspacecast %arg0 : !llvm.ptr<i32> to !llvm.ptr<i32, 2>
  llvm.return %1 : !llvm.ptr<i32, 2>
}

llvm.func @stringconstant() -> !llvm.ptr<i8> {
  %1 = llvm.mlir.constant("Hello world!") : !llvm.ptr<i8>
  // CHECK: ret [12 x i8] c"Hello world!"
  llvm.return %1 : !llvm.ptr<i8>
}

llvm.func @noreach() {
// CHECK:    unreachable
  llvm.unreachable
}

// CHECK-LABEL: define void @fcmp
llvm.func @fcmp(%arg0: !llvm.float, %arg1: !llvm.float) {
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
  %0 = llvm.fcmp "oeq" %arg0, %arg1 : !llvm.float
  %1 = llvm.fcmp "ogt" %arg0, %arg1 : !llvm.float
  %2 = llvm.fcmp "oge" %arg0, %arg1 : !llvm.float
  %3 = llvm.fcmp "olt" %arg0, %arg1 : !llvm.float
  %4 = llvm.fcmp "ole" %arg0, %arg1 : !llvm.float
  %5 = llvm.fcmp "one" %arg0, %arg1 : !llvm.float
  %6 = llvm.fcmp "ord" %arg0, %arg1 : !llvm.float
  %7 = llvm.fcmp "ueq" %arg0, %arg1 : !llvm.float
  %8 = llvm.fcmp "ugt" %arg0, %arg1 : !llvm.float
  %9 = llvm.fcmp "uge" %arg0, %arg1 : !llvm.float
  %10 = llvm.fcmp "ult" %arg0, %arg1 : !llvm.float
  %11 = llvm.fcmp "ule" %arg0, %arg1 : !llvm.float
  %12 = llvm.fcmp "une" %arg0, %arg1 : !llvm.float
  %13 = llvm.fcmp "uno" %arg0, %arg1 : !llvm.float
  llvm.return
}

// CHECK-LABEL: @vect
llvm.func @vect(%arg0: !llvm.vec<4 x float>, %arg1: !llvm.i32, %arg2: !llvm.float) {
  // CHECK-NEXT: extractelement <4 x float> {{.*}}, i32
  // CHECK-NEXT: insertelement <4 x float> {{.*}}, float %2, i32
  // CHECK-NEXT: shufflevector <4 x float> {{.*}}, <4 x float> {{.*}}, <5 x i32> <i32 0, i32 0, i32 0, i32 0, i32 7>
  %0 = llvm.extractelement %arg0[%arg1 : !llvm.i32] : !llvm.vec<4 x float>
  %1 = llvm.insertelement %arg2, %arg0[%arg1 : !llvm.i32] : !llvm.vec<4 x float>
  %2 = llvm.shufflevector %arg0, %arg0 [0 : i32, 0 : i32, 0 : i32, 0 : i32, 7 : i32] : !llvm.vec<4 x float>, !llvm.vec<4 x float>
  llvm.return
}

// CHECK-LABEL: @vect_i64idx
llvm.func @vect_i64idx(%arg0: !llvm.vec<4 x float>, %arg1: !llvm.i64, %arg2: !llvm.float) {
  // CHECK-NEXT: extractelement <4 x float> {{.*}}, i64
  // CHECK-NEXT: insertelement <4 x float> {{.*}}, float %2, i64
  %0 = llvm.extractelement %arg0[%arg1 : !llvm.i64] : !llvm.vec<4 x float>
  %1 = llvm.insertelement %arg2, %arg0[%arg1 : !llvm.i64] : !llvm.vec<4 x float>
  llvm.return
}

// CHECK-LABEL: @alloca
llvm.func @alloca(%size : !llvm.i64) {
  // Alignment automatically set by the LLVM IR builder when alignment attribute
  // is 0.
  //  CHECK: alloca {{.*}} align 4
  llvm.alloca %size x !llvm.i32 {alignment = 0} : (!llvm.i64) -> (!llvm.ptr<i32>)
  // CHECK-NEXT: alloca {{.*}} align 8
  llvm.alloca %size x !llvm.i32 {alignment = 8} : (!llvm.i64) -> (!llvm.ptr<i32>)
  llvm.return
}

// CHECK-LABEL: @constants
llvm.func @constants() -> !llvm.vec<4 x float> {
  // CHECK: ret <4 x float> <float 4.2{{0*}}e+01, float 0.{{0*}}e+00, float 0.{{0*}}e+00, float 0.{{0*}}e+00>
  %0 = llvm.mlir.constant(sparse<[[0]], [4.2e+01]> : vector<4xf32>) : !llvm.vec<4 x float>
  llvm.return %0 : !llvm.vec<4 x float>
}

// CHECK-LABEL: @fp_casts
llvm.func @fp_casts(%fp1 : !llvm.float, %fp2 : !llvm.double) -> !llvm.i16 {
// CHECK:    fptrunc double {{.*}} to float
  %a = llvm.fptrunc %fp2 : !llvm.double to !llvm.float
// CHECK:    fpext float {{.*}} to double
  %b = llvm.fpext %fp1 : !llvm.float to !llvm.double
// CHECK:    fptosi double {{.*}} to i16
  %c = llvm.fptosi %b : !llvm.double to !llvm.i16
  llvm.return %c : !llvm.i16
}

// CHECK-LABEL: @integer_extension_and_truncation
llvm.func @integer_extension_and_truncation(%a : !llvm.i32) {
// CHECK:    sext i32 {{.*}} to i64
// CHECK:    zext i32 {{.*}} to i64
// CHECK:    trunc i32 {{.*}} to i16
  %0 = llvm.sext %a : !llvm.i32 to !llvm.i64
  %1 = llvm.zext %a : !llvm.i32 to !llvm.i64
  %2 = llvm.trunc %a : !llvm.i32 to !llvm.i16
  llvm.return
}

// Check that the auxiliary `null` operation is converted into a `null` value.
// CHECK-LABEL: @null
llvm.func @null() -> !llvm.ptr<i32> {
  %0 = llvm.mlir.null : !llvm.ptr<i32>
  // CHECK: ret i32* null
  llvm.return %0 : !llvm.ptr<i32>
}

// Check that dense elements attributes are exported properly in constants.
// CHECK-LABEL: @elements_constant_3d_vector
llvm.func @elements_constant_3d_vector() -> !llvm.array<2 x array<2 x vec<2 x i32>>> {
  // CHECK: ret [2 x [2 x <2 x i32>]]
  // CHECK-SAME: {{\[}}[2 x <2 x i32>] [<2 x i32> <i32 1, i32 2>, <2 x i32> <i32 3, i32 4>],
  // CHECK-SAME:       [2 x <2 x i32>] [<2 x i32> <i32 42, i32 43>, <2 x i32> <i32 44, i32 45>]]
  %0 = llvm.mlir.constant(dense<[[[1, 2], [3, 4]], [[42, 43], [44, 45]]]> : vector<2x2x2xi32>) : !llvm.array<2 x array<2 x vec<2 x i32>>>
  llvm.return %0 : !llvm.array<2 x array<2 x vec<2 x i32>>>
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
    %f32_ptr : !llvm.ptr<float>, %f32 : !llvm.float,
    %i32_ptr : !llvm.ptr<i32>, %i32 : !llvm.i32) {
  // CHECK: atomicrmw fadd float* %{{.*}}, float %{{.*}} unordered
  %0 = llvm.atomicrmw fadd %f32_ptr, %f32 unordered : !llvm.float
  // CHECK: atomicrmw fsub float* %{{.*}}, float %{{.*}} unordered
  %1 = llvm.atomicrmw fsub %f32_ptr, %f32 unordered : !llvm.float
  // CHECK: atomicrmw xchg float* %{{.*}}, float %{{.*}} monotonic
  %2 = llvm.atomicrmw xchg %f32_ptr, %f32 monotonic : !llvm.float
  // CHECK: atomicrmw add i32* %{{.*}}, i32 %{{.*}} acquire
  %3 = llvm.atomicrmw add %i32_ptr, %i32 acquire : !llvm.i32
  // CHECK: atomicrmw sub i32* %{{.*}}, i32 %{{.*}} release
  %4 = llvm.atomicrmw sub %i32_ptr, %i32 release : !llvm.i32
  // CHECK: atomicrmw and i32* %{{.*}}, i32 %{{.*}} acq_rel
  %5 = llvm.atomicrmw _and %i32_ptr, %i32 acq_rel : !llvm.i32
  // CHECK: atomicrmw nand i32* %{{.*}}, i32 %{{.*}} seq_cst
  %6 = llvm.atomicrmw nand %i32_ptr, %i32 seq_cst : !llvm.i32
  // CHECK: atomicrmw or i32* %{{.*}}, i32 %{{.*}} unordered
  %7 = llvm.atomicrmw _or %i32_ptr, %i32 unordered : !llvm.i32
  // CHECK: atomicrmw xor i32* %{{.*}}, i32 %{{.*}} unordered
  %8 = llvm.atomicrmw _xor %i32_ptr, %i32 unordered : !llvm.i32
  // CHECK: atomicrmw max i32* %{{.*}}, i32 %{{.*}} unordered
  %9 = llvm.atomicrmw max %i32_ptr, %i32 unordered : !llvm.i32
  // CHECK: atomicrmw min i32* %{{.*}}, i32 %{{.*}} unordered
  %10 = llvm.atomicrmw min %i32_ptr, %i32 unordered : !llvm.i32
  // CHECK: atomicrmw umax i32* %{{.*}}, i32 %{{.*}} unordered
  %11 = llvm.atomicrmw umax %i32_ptr, %i32 unordered : !llvm.i32
  // CHECK: atomicrmw umin i32* %{{.*}}, i32 %{{.*}} unordered
  %12 = llvm.atomicrmw umin %i32_ptr, %i32 unordered : !llvm.i32
  llvm.return
}

// CHECK-LABEL: @cmpxchg
llvm.func @cmpxchg(%ptr : !llvm.ptr<float>, %cmp : !llvm.float, %val: !llvm.float) {
  // CHECK: cmpxchg float* %{{.*}}, float %{{.*}}, float %{{.*}} acq_rel monotonic
  %0 = llvm.cmpxchg %ptr, %cmp, %val acq_rel monotonic : !llvm.float
  // CHECK: %{{[0-9]+}} = extractvalue { float, i1 } %{{[0-9]+}}, 0
  %1 = llvm.extractvalue %0[0] : !llvm.struct<(float, i1)>
  // CHECK: %{{[0-9]+}} = extractvalue { float, i1 } %{{[0-9]+}}, 1
  %2 = llvm.extractvalue %0[1] : !llvm.struct<(float, i1)>
  llvm.return
}

llvm.mlir.global external constant @_ZTIi() : !llvm.ptr<i8>
llvm.func @foo(!llvm.ptr<i8>)
llvm.func @bar(!llvm.ptr<i8>) -> !llvm.ptr<i8>
llvm.func @__gxx_personality_v0(...) -> !llvm.i32

// CHECK-LABEL: @invokeLandingpad
llvm.func @invokeLandingpad() -> !llvm.i32 attributes { personality = @__gxx_personality_v0 } {
// CHECK: %[[a1:[0-9]+]] = alloca i8
  %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
  %1 = llvm.mlir.constant("\01") : !llvm.array<1 x i8>
  %2 = llvm.mlir.addressof @_ZTIi : !llvm.ptr<ptr<i8>>
  %3 = llvm.bitcast %2 : !llvm.ptr<ptr<i8>> to !llvm.ptr<i8>
  %4 = llvm.mlir.null : !llvm.ptr<ptr<i8>>
  %5 = llvm.mlir.constant(1 : i32) : !llvm.i32
  %6 = llvm.alloca %5 x !llvm.i8 : (!llvm.i32) -> !llvm.ptr<i8>
// CHECK: invoke void @foo(i8* %[[a1]])
// CHECK-NEXT: to label %[[normal:[0-9]+]] unwind label %[[unwind:[0-9]+]]
  llvm.invoke @foo(%6) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>) -> ()

// CHECK: [[unwind]]:
^bb1:
// CHECK: %{{[0-9]+}} = landingpad { i8*, i32 }
// CHECK-NEXT:             catch i8** null
// CHECK-NEXT:             catch i8* bitcast (i8** @_ZTIi to i8*)
// CHECK-NEXT:             filter [1 x i8] c"\01"
  %7 = llvm.landingpad (catch %4 : !llvm.ptr<ptr<i8>>) (catch %3 : !llvm.ptr<i8>) (filter %1 : !llvm.array<1 x i8>) : !llvm.struct<(ptr<i8>, i32)>
// CHECK: br label %[[final:[0-9]+]]
  llvm.br ^bb3

// CHECK: [[normal]]:
// CHECK-NEXT: ret i32 1
^bb2:	// 2 preds: ^bb0, ^bb3
  llvm.return %5 : !llvm.i32

// CHECK: [[final]]:
// CHECK-NEXT: %{{[0-9]+}} = invoke i8* @bar(i8* %[[a1]])
// CHECK-NEXT:          to label %[[normal]] unwind label %[[unwind]]
^bb3:	// pred: ^bb1
  %8 = llvm.invoke @bar(%6) to ^bb2 unwind ^bb1 : (!llvm.ptr<i8>) -> !llvm.ptr<i8>
}

// CHECK-LABEL: @callFreezeOp
llvm.func @callFreezeOp(%x : !llvm.i32) {
  // CHECK: freeze i32 %{{[0-9]+}}
  %0 = llvm.freeze %x : !llvm.i32
  %1 = llvm.mlir.undef : !llvm.i32
  // CHECK: freeze i32 undef
  %2 = llvm.freeze %1 : !llvm.i32
  llvm.return
}

// CHECK-LABEL: @boolConstArg
llvm.func @boolConstArg() -> !llvm.i1 {
  // CHECK: ret i1 false
  %0 = llvm.mlir.constant(true) : !llvm.i1
  %1 = llvm.mlir.constant(false) : !llvm.i1
  %2 = llvm.and %0, %1 : !llvm.i1
  llvm.return %2 : !llvm.i1
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
llvm.func @passthrough() attributes {passthrough = ["noinline", ["alignstack", "4"], "null_pointer_is_valid", ["foo", "bar"]]} {
  llvm.return
}

// CHECK: attributes #[[ATTR_GROUP]] = {
// CHECK-DAG: noinline
// CHECK-DAG: alignstack=4
// CHECK-DAG: null_pointer_is_valid
// CHECK-DAG: "foo"="bar"

// -----

// CHECK-LABEL: @constant_bf16
llvm.func @constant_bf16() -> !llvm.bfloat {
  %0 = llvm.mlir.constant(1.000000e+01 : bf16) : !llvm.bfloat
  llvm.return %0 : !llvm.bfloat
}

// CHECK: ret bfloat 0xR4120

// -----

llvm.func @address_taken() {
  llvm.return
}

llvm.mlir.global internal constant @taker_of_address() : !llvm.ptr<func<void ()>> {
  %0 = llvm.mlir.addressof @address_taken : !llvm.ptr<func<void ()>>
  llvm.return %0 : !llvm.ptr<func<void ()>>
}

// -----

// Check that branch weight attributes are exported properly as metadata.
llvm.func @cond_br_weights(%cond : !llvm.i1, %arg0 : !llvm.i32,  %arg1 : !llvm.i32) -> !llvm.i32 {
  // CHECK: !prof ![[NODE:[0-9]+]]
  llvm.cond_br %cond weights(dense<[5, 10]> : vector<2xi32>), ^bb1, ^bb2
^bb1:  // pred: ^bb0
  llvm.return %arg0 : !llvm.i32
^bb2:  // pred: ^bb0
  llvm.return %arg1 : !llvm.i32
}

// CHECK: ![[NODE]] = !{!"branch_weights", i32 5, i32 10}

// -----

llvm.func @volatile_store_and_load() {
  %val = llvm.mlir.constant(5 : i32) : !llvm.i32
  %size = llvm.mlir.constant(1 : i64) : !llvm.i64
  %0 = llvm.alloca %size x !llvm.i32 : (!llvm.i64) -> (!llvm.ptr<i32>)
  // CHECK: store volatile i32 5, i32* %{{.*}}
  llvm.store volatile %val, %0 : !llvm.ptr<i32>
  // CHECK: %{{.*}} = load volatile i32, i32* %{{.*}}
  %1 = llvm.load volatile %0: !llvm.ptr<i32>
  llvm.return
}

// -----

// Check that nontemporal attribute is exported as metadata node.
llvm.func @nontemoral_store_and_load() {
  %val = llvm.mlir.constant(5 : i32) : !llvm.i32
  %size = llvm.mlir.constant(1 : i64) : !llvm.i64
  %0 = llvm.alloca %size x !llvm.i32 : (!llvm.i64) -> (!llvm.ptr<i32>)
  // CHECK: !nontemporal ![[NODE:[0-9]+]]
  llvm.store %val, %0 {nontemporal} : !llvm.ptr<i32>
  // CHECK: !nontemporal ![[NODE]]
  %1 = llvm.load %0 {nontemporal} : !llvm.ptr<i32>
  llvm.return
}

// CHECK: ![[NODE]] = !{i32 1}
