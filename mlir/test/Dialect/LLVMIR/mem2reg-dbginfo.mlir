// RUN: mlir-opt %s --pass-pipeline='builtin.module(llvm.func(mem2reg{region-simplify=false}))' | FileCheck %s

llvm.func @use(i64)
llvm.func @use_ptr(!llvm.ptr)

#di_basic_type = #llvm.di_basic_type<tag = DW_TAG_base_type, name = "ptr sized type", sizeInBits = 64>
#di_file = #llvm.di_file<"test.ll" in "">
#di_compile_unit = #llvm.di_compile_unit<sourceLanguage = DW_LANG_C_plus_plus_14, file = #di_file, producer = "clang", isOptimized = false, emissionKind = Full>
#di_subprogram = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "blah", linkageName = "blah", file = #di_file, line = 7, subprogramFlags = Definition>
// CHECK: #[[$VAR:.*]] = #llvm.di_local_variable<{{.*}}name = "ptr sized var"{{.*}}>
#di_local_variable = #llvm.di_local_variable<scope = #di_subprogram, name = "ptr sized var", file = #di_file, line = 7, arg = 1, type = #di_basic_type>
#di_local_variable_2 = #llvm.di_local_variable<scope = #di_subprogram, name = "ptr sized var 2", file = #di_file, line = 7, arg = 1, type = #di_basic_type>

// CHECK-LABEL: llvm.func @basic_store_load
llvm.func @basic_store_load(%arg0: i64) -> i64 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NOT: = llvm.alloca
  %1 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK-NOT: llvm.store
  llvm.store %arg0, %1 {alignment = 4 : i64} : i64, !llvm.ptr
  // CHECK-NOT: llvm.intr.dbg.declare
  llvm.intr.dbg.declare #di_local_variable = %1 : !llvm.ptr
  // CHECK: llvm.intr.dbg.value #[[$VAR]] = %[[LOADED:.*]] : i64
  // CHECK-NOT: llvm.intr.dbg.value
  // CHECK-NOT: llvm.intr.dbg.declare
  // CHECK-NOT: llvm.store
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i64
  // CHECK: llvm.return %[[LOADED]] : i64
  llvm.return %2 : i64
}

// CHECK-LABEL: llvm.func @block_argument_value
// CHECK-SAME: (%[[ARG0:.*]]: i64, {{.*}})
llvm.func @block_argument_value(%arg0: i64, %arg1: i1) -> i64 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NOT: = llvm.alloca
  %1 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK-NOT: llvm.intr.dbg.declare
  llvm.intr.dbg.declare #di_local_variable = %1 : !llvm.ptr
  llvm.cond_br %arg1, ^bb1, ^bb2
// CHECK: ^{{.*}}:
^bb1:
  // CHECK: llvm.intr.dbg.value #[[$VAR]] = %[[ARG0]]
  // CHECK-NOT: llvm.intr.dbg.value
  llvm.store %arg0, %1 {alignment = 4 : i64} : i64, !llvm.ptr
  llvm.br ^bb2
// CHECK: ^{{.*}}(%[[BLOCKARG:.*]]: i64):
^bb2:
  // CHECK: llvm.intr.dbg.value #[[$VAR]] = %[[BLOCKARG]]
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i64
  llvm.return %2 : i64
}

// CHECK-LABEL: llvm.func @double_block_argument_value
// CHECK-SAME: (%[[ARG0:.*]]: i64, {{.*}})
llvm.func @double_block_argument_value(%arg0: i64, %arg1: i1) -> i64 {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NOT: = llvm.alloca
  %1 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK-NOT: llvm.intr.dbg.declare
  llvm.intr.dbg.declare #di_local_variable = %1 : !llvm.ptr
  llvm.cond_br %arg1, ^bb1, ^bb2
// CHECK: ^{{.*}}(%[[BLOCKARG1:.*]]: i64):
^bb1:
  // CHECK: llvm.intr.dbg.value #[[$VAR]] = %[[BLOCKARG1]]
  %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i64
  llvm.call @use(%2) : (i64) -> ()
  // CHECK: llvm.intr.dbg.value #[[$VAR]] = %[[ARG0]]
  llvm.store %arg0, %1 {alignment = 4 : i64} : i64, !llvm.ptr
  llvm.br ^bb2
  // CHECK-NOT: llvm.intr.dbg.value
// CHECK: ^{{.*}}(%[[BLOCKARG2:.*]]: i64):
^bb2:
  // CHECK: llvm.intr.dbg.value #[[$VAR]] = %[[BLOCKARG2]]
  llvm.br ^bb1
}

// CHECK-LABEL: llvm.func @always_drop_promoted_declare
// CHECK: %[[UNDEF:.*]] = llvm.mlir.undef
// CHECK-NOT: = llvm.alloca
// CHECK-NOT: llvm.intr.dbg.declare
// CHECK: llvm.intr.dbg.value #{{.*}} = %[[UNDEF]]
llvm.func @always_drop_promoted_declare() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.intr.dbg.declare #di_local_variable = %1 : !llvm.ptr
  llvm.intr.dbg.value #di_local_variable = %1 : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: llvm.func @keep_dbg_if_not_promoted
llvm.func @keep_dbg_if_not_promoted() {
  %0 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: %[[ALLOCA:.*]] = llvm.alloca
  %1 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  // CHECK-NOT: = llvm.alloca
  // CHECK-NOT: llvm.intr.dbg.declare
  // CHECK: llvm.intr.dbg.declare #[[$VAR]] = %[[ALLOCA]]
  // CHECK-NOT: = llvm.alloca
  // CHECK-NOT: llvm.intr.dbg.declare
  // CHECK: llvm.call @use_ptr(%[[ALLOCA]])
  llvm.intr.dbg.declare #di_local_variable = %1 : !llvm.ptr
  %2 = llvm.alloca %0 x i64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.intr.dbg.declare #di_local_variable_2 = %2 : !llvm.ptr
  llvm.call @use_ptr(%1) : (!llvm.ptr) -> ()
  llvm.return
}
