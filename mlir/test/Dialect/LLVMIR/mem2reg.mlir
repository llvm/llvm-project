// RUN: mlir-opt %s --pass-pipeline="builtin.module(llvm.func(mem2reg{region-simplify=false}))" --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @default_value
llvm.func @default_value() -> i32 {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.return %[[UNDEF]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK-LABEL: llvm.func @store_of_ptr
llvm.func @store_of_ptr() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(4 : i32) : i32
  %2 = llvm.mlir.zero : !llvm.ptr
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca
  %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK: llvm.store %{{.*}}, %[[ALLOCA]]
  llvm.store %1, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  // CHECK: llvm.store %[[ALLOCA]], %{{.*}}
  llvm.store %3, %2 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @unreachable
llvm.func @unreachable() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  // CHECK-NOT: = llvm.alloca
  %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.return

// CHECK: ^{{.*}}:
// CHECK-NEXT: llvm.return
^bb1:  // no predecessors
  llvm.store %1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @unreachable_in_loop
// CHECK-NOT: = llvm.alloca
llvm.func @unreachable_in_loop() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(6 : i32) : i32
  %2 = llvm.mlir.constant(5 : i32) : i32
  %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %1, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  // CHECK: llvm.br ^[[LOOP:.*]]
  llvm.br ^bb1

// CHECK: ^[[LOOP]]:
^bb1:  // 2 preds: ^bb0, ^bb3
  // CHECK-NEXT: llvm.br ^[[ENDOFLOOP:.*]]
  llvm.store %2, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.br ^bb3

// CHECK: ^[[UNREACHABLE:.*]]:
^bb2:  // no predecessors
  // CHECK-NEXT: llvm.br ^[[ENDOFLOOP]]
  llvm.br ^bb3

// CHECK: ^[[ENDOFLOOP]]:
^bb3:  // 2 preds: ^bb1, ^bb2
  // CHECK-NEXT: llvm.br ^[[LOOP]]
  llvm.br ^bb1
}

// -----

// CHECK-LABEL: llvm.func @branching
// CHECK-NOT: = llvm.alloca
llvm.func @branching(%arg0: i1, %arg1: i1) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK: llvm.cond_br %{{.*}}, ^[[BB2:.*]](%{{.*}} : i32), ^{{.*}}
  llvm.cond_br %arg0, ^bb2, ^bb1
^bb1:  // pred: ^bb0
  llvm.store %1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
  // CHECK: llvm.cond_br %{{.*}}, ^[[BB2]](%{{.*}} : i32), ^[[BB2]](%{{.*}} : i32)
  llvm.cond_br %arg1, ^bb2, ^bb2
// CHECK: ^[[BB2]](%[[V3:.*]]: i32):
^bb2:  // 3 preds: ^bb0, ^bb1, ^bb1
  %3 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.return %[[V3]] : i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @recursive_alloca
// CHECK-NOT: = llvm.alloca
llvm.func @recursive_alloca() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %1, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.store %3, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  %5 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  %6 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.store %6, %2 {alignment = 4 : i64} : i32, !llvm.ptr
  %7 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.return %7 : i32
}

// -----

// CHECK-LABEL: llvm.func @reset_in_branch
// CHECK-NOT: = llvm.alloca
// CHECK-NOT: ^{{.*}}({{.*}}):
llvm.func @reset_in_branch(%arg0: i32, %arg1: i1) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(true) : i1
  %2 = llvm.mlir.constant(false) : i1
  %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.cond_br %arg1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  llvm.store %arg0, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  %4 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.call @reset_in_branch(%4, %2) : (i32, i1) -> ()
  llvm.br ^bb3
^bb2:  // pred: ^bb0
  %5 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.call @reset_in_branch(%5, %1) : (i32, i1) -> ()
  llvm.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @intertwined_alloca
// CHECK-NOT: = llvm.alloca
llvm.func @intertwined_alloca(%arg0: !llvm.ptr, %arg1: i32) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  %2 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %2 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.store %arg1, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.store %1, %4 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb4
  %7 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
  %8 = llvm.add %7, %0  : i32
  %9 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
  %10 = llvm.icmp "sgt" %8, %9 : i32
  %11 = llvm.zext %10 : i1 to i32
  llvm.cond_br %10, ^bb2, ^bb5
^bb2:  // pred: ^bb1
  %12 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.store %12, %5 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.store %1, %6 {alignment = 4 : i64} : i32, !llvm.ptr
  %13 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
  %14 = llvm.icmp "sgt" %13, %1 : i32
  %15 = llvm.zext %14 : i1 to i32
  llvm.cond_br %14, ^bb3, ^bb4
^bb3:  // pred: ^bb2
  %16 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  %17 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
  %18 = llvm.sub %17, %0  : i32
  %19 = llvm.getelementptr %16[%18] : (!llvm.ptr, i32) -> !llvm.ptr, i8
  %20 = llvm.load %5 {alignment = 4 : i64} : !llvm.ptr -> i32
  %21 = llvm.trunc %20 : i32 to i8
  llvm.store %21, %19 {alignment = 1 : i64} : i8, !llvm.ptr
  llvm.br ^bb4
^bb4:  // 2 preds: ^bb2, ^bb3
  %22 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
  %23 = llvm.add %22, %0  : i32
  llvm.store %23, %4 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.br ^bb1
^bb5:  // pred: ^bb1
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @complex_cf
// CHECK-NOT: = llvm.alloca
llvm.func @complex_cf(%arg0: i32, ...) {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(false) : i1
  %2 = llvm.mlir.constant(0 : i32) : i32
  %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.cond_br %1, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  llvm.br ^bb2
^bb2:  // 2 preds: ^bb0, ^bb1
  llvm.store %2, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.br ^bb3
^bb3:  // 2 preds: ^bb2, ^bb16
  llvm.cond_br %1, ^bb4, ^bb17
^bb4:  // pred: ^bb3
  llvm.cond_br %1, ^bb5, ^bb14
^bb5:  // pred: ^bb4
  llvm.cond_br %1, ^bb7, ^bb6
^bb6:  // pred: ^bb5
  llvm.br ^bb7
^bb7:  // 2 preds: ^bb5, ^bb6
  llvm.cond_br %1, ^bb9, ^bb8
^bb8:  // pred: ^bb7
  llvm.br ^bb9
^bb9:  // 2 preds: ^bb7, ^bb8
  llvm.cond_br %1, ^bb11, ^bb10
^bb10:  // pred: ^bb9
  llvm.br ^bb11
^bb11:  // 2 preds: ^bb9, ^bb10
  llvm.cond_br %1, ^bb12, ^bb13
^bb12:  // pred: ^bb11
  llvm.br ^bb13
^bb13:  // 2 preds: ^bb11, ^bb12
  llvm.br ^bb14
^bb14:  // 2 preds: ^bb4, ^bb13
  llvm.cond_br %1, ^bb15, ^bb16
^bb15:  // pred: ^bb14
  llvm.br ^bb16
^bb16:  // 2 preds: ^bb14, ^bb15
  llvm.br ^bb3
^bb17:  // pred: ^bb3
  llvm.br ^bb20
^bb18:  // no predecessors
  %4 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.br ^bb24
^bb19:  // no predecessors
  llvm.br ^bb20
^bb20:  // 2 preds: ^bb17, ^bb19
  llvm.cond_br %1, ^bb21, ^bb22
^bb21:  // pred: ^bb20
  llvm.br ^bb23
^bb22:  // pred: ^bb20
  llvm.br ^bb23
^bb23:  // 2 preds: ^bb21, ^bb22
  llvm.br ^bb24
^bb24:  // 2 preds: ^bb18, ^bb23
  llvm.br ^bb26
^bb25:  // no predecessors
  llvm.br ^bb26
^bb26:  // 2 preds: ^bb24, ^bb25
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @llvm_crash
llvm.func @llvm_crash() -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  %2 = llvm.mlir.addressof @j : !llvm.ptr
  %3 = llvm.mlir.constant(0 : i8) : i8
  // CHECK-NOT: = llvm.alloca
  // CHECK: %[[VOLATILE_ALLOCA:.*]] = llvm.alloca
  // CHECK-NOT: = llvm.alloca
  %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %6 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %7 = llvm.bitcast %1 : i32 to i32
  // CHECK: llvm.store volatile %{{.*}}, %[[VOLATILE_ALLOCA]]
  llvm.store volatile %1, %5 {alignment = 4 : i64} : i32, !llvm.ptr
  %8 = llvm.call @_setjmp(%2) : (!llvm.ptr) -> i32
  %9 = llvm.icmp "ne" %8, %1 : i32
  %10 = llvm.zext %9 : i1 to i8
  %11 = llvm.icmp "ne" %10, %3 : i8
  llvm.cond_br %11, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  // CHECK: = llvm.load volatile %[[VOLATILE_ALLOCA]]
  %12 = llvm.load volatile %5 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.store %12, %6 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.br ^bb3
^bb2:  // pred: ^bb0
  // CHECK: llvm.store volatile %{{.*}}, %[[VOLATILE_ALLOCA]]
  llvm.store volatile %0, %5 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.call @g() : () -> ()
  llvm.store %1, %6 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %13 = llvm.load %6 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.store %13, %4 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.br ^bb4
^bb4:  // pred: ^bb3
  %14 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.return %14 : i32
}
llvm.mlir.global external @j() {addr_space = 0 : i32} : !llvm.array<1 x struct<"struct.__jmp_buf_tag", (array<6 x i32>, i32, struct<"struct.__sigset_t", (array<32 x i32>)>)>>
llvm.func @_setjmp(!llvm.ptr) -> i32 attributes {passthrough = ["returns_twice"]}
llvm.func @g()

// -----

// CHECK-LABEL: llvm.func amdgpu_kernelcc @addrspace_discard
// CHECK-NOT: = llvm.alloca
llvm.func amdgpu_kernelcc @addrspace_discard() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i64) : i64
  %2 = llvm.alloca %0 x i8 {alignment = 8 : i64} : (i32) -> !llvm.ptr<5>
  %3 = llvm.addrspacecast %2 : !llvm.ptr<5> to !llvm.ptr
  llvm.intr.lifetime.start 2, %3 : !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @ignore_atomic
// CHECK-SAME: (%[[ARG0:.*]]: i32) -> i32
llvm.func @ignore_atomic(%arg0: i32) -> i32 {
  // CHECK-NOT: = llvm.alloca
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %1 atomic seq_cst {alignment = 4 : i64} : i32, !llvm.ptr
  %2 = llvm.load %1 atomic seq_cst {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.return %[[ARG0]] : i32
  llvm.return %2 : i32
}

// -----

// CHECK: llvm.func @landing_pad
// CHECK-NOT: = llvm.alloca
llvm.func @landing_pad() -> i32 attributes {personality = @__gxx_personality_v0} {
  // CHECK-NOT: = llvm.alloca
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : i32
  // CHECK-NOT: = llvm.alloca
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK: %[[V2:.*]] = llvm.invoke
  %2 = llvm.invoke @landing_padf() to ^bb1 unwind ^bb3 : () -> i32
// CHECK: ^{{.*}}:
^bb1:// pred: ^bb0
  llvm.store %2, %1 {alignment = 4 : i64} : i32, !llvm.ptr
  // CHECK: llvm.br ^[[BB2:.*]](%[[V2]] : i32)
  llvm.br ^bb2
// CHECK: ^[[BB2]]([[V3:.*]]: i32):
^bb2:// 2 preds: ^bb1, ^bb3
  %3 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.return [[V3]] : i32
  llvm.return %3 : i32
// CHECK: ^{{.*}}:
^bb3:// pred: ^bb0
  %4 = llvm.landingpad cleanup : !llvm.struct<(ptr, i32)>
  // CHECK: llvm.br ^[[BB2:.*]](%[[UNDEF]] : i32)
  llvm.br ^bb2
}
llvm.func @landing_padf() -> i32
llvm.func @__gxx_personality_v0(...) -> i32

// -----

// CHECK-LABEL: llvm.func @unreachable_defines
llvm.func @unreachable_defines() -> i32 {
  // CHECK-NOT: = llvm.alloca
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : i32
  // CHECK-NOT: = llvm.alloca
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.br ^bb1
^bb1:  // 2 preds: ^bb0, ^bb2
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.return %[[UNDEF]] : i32
  llvm.return %2 : i32
^bb2:  // no predecessors
  %3 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.store %3, %1 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.br ^bb1
}

// -----

// CHECK-LABEL: llvm.func @unreachable_jumps_to_merge_point
// CHECK-NOT: = llvm.alloca
llvm.func @unreachable_jumps_to_merge_point(%arg0: i1) -> i32 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(6 : i32) : i32
  %2 = llvm.mlir.constant(5 : i32) : i32
  %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.cond_br %arg0, ^bb1, ^bb2
^bb1:  // 2 preds: ^bb0, ^bb4
  llvm.store %1, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.br ^bb4
^bb2:  // pred: ^bb0
  llvm.store %2, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.br ^bb4
^bb3:  // no predecessors
  llvm.br ^bb4
^bb4:  // 3 preds: ^bb1, ^bb2, ^bb3
  %4 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.return %4 : i32
}

// -----

// CHECK-LABEL: llvm.func @ignore_lifetime
// CHECK-NOT: = llvm.alloca
llvm.func @ignore_lifetime() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.intr.lifetime.start 2, %1 : !llvm.ptr
  llvm.store %0, %1 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.intr.lifetime.end 2, %1 : !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @ignore_discardable_tree
// CHECK-NOT: = llvm.alloca
llvm.func @ignore_discardable_tree() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i16) : i16
  %2 = llvm.mlir.constant(0 : i8) : i8
  %3 = llvm.mlir.undef : !llvm.struct<(i8, i16)>
  %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(i8, i16)>
  %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(i8, i16)>
  %6 = llvm.alloca %0 x !llvm.struct<(i8, i16)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %7 = llvm.getelementptr %6[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i8, i16)>
  llvm.intr.lifetime.start 2, %7 : !llvm.ptr
  llvm.store %5, %6 {alignment = 2 : i64} : !llvm.struct<(i8, i16)>, !llvm.ptr
  llvm.intr.lifetime.end 2, %7 : !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @store_load_forward
llvm.func @store_load_forward() -> i32 {
  // CHECK-NOT: = llvm.alloca
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[RES:.*]] = llvm.mlir.constant(0 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
  %3 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: llvm.return %[[RES]] : i32
  llvm.return %3 : i32
}

// -----

// CHECK-LABEL: llvm.func @store_load_wrong_type
llvm.func @store_load_wrong_type() -> i16 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: = llvm.alloca
  %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
  %3 = llvm.load %2 {alignment = 2 : i64} : !llvm.ptr -> i16
  llvm.return %3 : i16
}

// -----

// CHECK-LABEL: llvm.func @merge_point_cycle
llvm.func @merge_point_cycle() {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(7 : i32) : i32
  %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK: llvm.br ^[[BB1:.*]](%[[UNDEF]] : i32)
  llvm.br ^bb1
// CHECK: ^[[BB1]](%[[BARG:.*]]: i32):
^bb1:  // 2 preds: ^bb0, ^bb1
  %3 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: = llvm.call @use(%[[BARG]])
  %4 = llvm.call @use(%3) : (i32) -> i1
  // CHECK: %[[DEF:.*]] = llvm.call @def
  %5 = llvm.call @def(%1) : (i32) -> i32
  llvm.store %5, %2 {alignment = 4 : i64} : i32, !llvm.ptr
  // CHECK: llvm.cond_br %{{.*}}, ^[[BB1]](%[[DEF]] : i32), ^{{.*}}
  llvm.cond_br %4, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  llvm.return
}

llvm.func @def(i32) -> i32
llvm.func @use(i32) -> i1

// -----

// CHECK-LABEL: llvm.func @no_unnecessary_arguments
llvm.func @no_unnecessary_arguments() {
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : i32
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  // CHECK: llvm.br ^[[BB1:.*]]
  llvm.br ^bb1
// CHECK: ^[[BB1]]:
^bb1:  // 2 preds: ^bb0, ^bb1
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
  // CHECK: = llvm.call @use(%[[UNDEF]])
  %3 = llvm.call @use(%2) : (i32) -> i1
  // CHECK: llvm.cond_br %{{.*}}, ^[[BB1]], ^{{.*}}
  llvm.cond_br %3, ^bb1, ^bb2
^bb2:  // pred: ^bb1
  llvm.return
}

llvm.func @use(i32) -> i1

// -----

// CHECK-LABEL: llvm.func @discardable_use_tree
// CHECK-NOT: = llvm.alloca
llvm.func @discardable_use_tree() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i64) : i64
  %2 = llvm.alloca %0 x i8 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.bitcast %2 : !llvm.ptr to !llvm.ptr
  %4 = llvm.bitcast %3 : !llvm.ptr to !llvm.ptr
  llvm.intr.lifetime.start 2, %3 : !llvm.ptr
  llvm.intr.lifetime.start 2, %4 : !llvm.ptr
  %5 = llvm.intr.invariant.start 2, %3 : !llvm.ptr
  llvm.intr.invariant.end %5, 2, %3 : !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @non_discardable_use_tree
llvm.func @non_discardable_use_tree() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i64) : i64
  // CHECK: = llvm.alloca
  %2 = llvm.alloca %0 x i8 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.bitcast %2 : !llvm.ptr to !llvm.ptr
  %4 = llvm.bitcast %3 : !llvm.ptr to !llvm.ptr
  llvm.intr.lifetime.start 2, %3 : !llvm.ptr
  llvm.intr.lifetime.start 2, %4 : !llvm.ptr
  llvm.call @use(%4) : (!llvm.ptr) -> i1
  llvm.return
}
llvm.func @use(!llvm.ptr) -> i1

// -----

// CHECK-LABEL: llvm.func @trivial_get_element_ptr
// CHECK-NOT: = llvm.alloca
llvm.func @trivial_get_element_ptr() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i64) : i64
  %2 = llvm.alloca %0 x i8 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.bitcast %2 : !llvm.ptr to !llvm.ptr
  %4 = llvm.getelementptr %3[0] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.intr.lifetime.start 2, %3 : !llvm.ptr
  llvm.intr.lifetime.start 2, %4 : !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @nontrivial_get_element_ptr
llvm.func @nontrivial_get_element_ptr() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i64) : i64
  // CHECK: = llvm.alloca
  %2 = llvm.alloca %0 x i8 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.getelementptr %2[1] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.intr.lifetime.start 2, %2 : !llvm.ptr
  llvm.intr.lifetime.start 2, %4 : !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @dynamic_get_element_ptr
llvm.func @dynamic_get_element_ptr() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i64) : i64
  // CHECK: = llvm.alloca
  %2 = llvm.alloca %0 x i8 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.bitcast %2 : !llvm.ptr to !llvm.ptr
  %4 = llvm.getelementptr %3[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i8
  llvm.intr.lifetime.start 2, %3 : !llvm.ptr
  llvm.intr.lifetime.start 2, %4 : !llvm.ptr
  llvm.return
}

// -----

// CHECK-LABEL: llvm.func @live_cycle
// CHECK-SAME: (%[[ARG0:.*]]: i64, %{{.*}}: i1, %[[ARG2:.*]]: i64) -> i64
llvm.func @live_cycle(%arg0: i64, %arg1: i1, %arg2: i64) -> i64 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NOT: = llvm.alloca
  %1 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg2, %1 {alignment = 4 : i64} : i64, !llvm.ptr
  // CHECK: llvm.cond_br %{{.*}}, ^[[BB1:.*]](%[[ARG2]] : i64), ^[[BB2:.*]](%[[ARG2]] : i64)
  llvm.cond_br %arg1, ^bb1, ^bb2
// CHECK: ^[[BB1]](%[[V1:.*]]: i64):
^bb1:
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i64
  // CHECK: llvm.call @use(%[[V1]])
  llvm.call @use(%2) : (i64) -> ()
  llvm.store %arg0, %1 {alignment = 4 : i64} : i64, !llvm.ptr
  // CHECK: llvm.br ^[[BB2]](%[[ARG0]] : i64)
  llvm.br ^bb2
// CHECK: ^[[BB2]](%[[V2:.*]]: i64):
^bb2:
  // CHECK: llvm.br ^[[BB1]](%[[V2]] : i64)
  llvm.br ^bb1
}

llvm.func @use(i64)

// -----

// This test should no longer be an issue once promotion within subregions
// is supported.
// CHECK-LABEL: llvm.func @subregion_block_promotion
// CHECK-SAME: (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64) -> i64
llvm.func @subregion_block_promotion(%arg0: i64, %arg1: i64) -> i64 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca
  %1 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK: llvm.store %[[ARG1]], %[[ALLOCA]]
  llvm.store %arg1, %1 {alignment = 4 : i64} : i64, !llvm.ptr
  // CHECK: scf.execute_region {
  scf.execute_region {
    // CHECK: llvm.store %[[ARG0]], %[[ALLOCA]]
    llvm.store %arg0, %1 {alignment = 4 : i64} : i64, !llvm.ptr
    scf.yield
  }
  // CHECK: }
  // CHECK: %[[RES:.*]] = llvm.load %[[ALLOCA]]
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i64
  // CHECK: llvm.return %[[RES]] : i64
  llvm.return %2 : i64
}

// -----

// CHECK-LABEL: llvm.func @subregion_simple_transitive_promotion
// CHECK-SAME: (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64) -> i64
llvm.func @subregion_simple_transitive_promotion(%arg0: i64, %arg1: i64) -> i64 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NOT: = llvm.alloca
  %1 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg1, %1 {alignment = 4 : i64} : i64, !llvm.ptr
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i64
  // CHECK: scf.execute_region {
  scf.execute_region {
    // CHECK: llvm.call @use(%[[ARG1]])
    llvm.call @use(%2) : (i64) -> ()
    scf.yield
  }
  // CHECK: }
  // CHECK: llvm.return %[[ARG1]] : i64
  llvm.return %2 : i64
}

llvm.func @use(i64)

// -----

// This behavior is specific to the LLVM dialect, because LLVM semantics are
// that reaching an alloca multiple times allocates on the stack multiple
// times. Promoting an alloca that is reached multiple times could lead to
// changes in observable behavior. Thus only allocas in the entry block are
// promoted.

// CHECK-LABEL: llvm.func @no_inner_alloca_promotion
// CHECK-SAME: (%[[ARG:.*]]: i64) -> i64
llvm.func @no_inner_alloca_promotion(%arg: i64) -> i64 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  llvm.br ^bb1
^bb1:
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca
  %1 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK: llvm.store %[[ARG]], %[[ALLOCA]]
  llvm.store %arg, %1 {alignment = 4 : i64} : i64, !llvm.ptr
  // CHECK: %[[RES:.*]] = llvm.load %[[ALLOCA]]
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i64
  // CHECK: llvm.return %[[RES]] : i64
  llvm.return %2 : i64
}

// -----

// CHECK-LABEL: @transitive_reaching_def
llvm.func @transitive_reaching_def() -> !llvm.ptr {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NOT: alloca
  %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %2 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  llvm.store %2, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  %3 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  llvm.return %3 : !llvm.ptr
}
