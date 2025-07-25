// RUN: mlir-translate --split-input-file -mlir-to-llvmir %s | FileCheck %s

module {
  llvm.func spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
  llvm.func @blockload2d_cache_control(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> vector<8xi16> {
    %0 = llvm.mlir.undef : vector<2xi32>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.insertelement %arg4, %0[%1 : i32] : vector<2xi32>
    %4 = llvm.insertelement %arg5, %3[%2 : i32] : vector<2xi32>
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.alloca %5 x i16 : (i32) -> !llvm.ptr
    // CHECK-LABEL: call spir_func void @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt
    // CHECK-SAME: !spirv.DecorationCacheControlINTEL ![[DECO1:.*]]
    llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(%arg0, %arg1, %arg2, %arg3, %4, %6)
      {function_type = !llvm.func<void (ptr<1>, i32, i32, i32, vector<2xi32>, ptr)>, linkage = #llvm.linkage<external>, no_unwind,
       sym_name = "_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt", visibility_ = 0 : i64, will_return,
       xevm.DecorationCacheControl = [[6442 : i32, 0 : i32, 1 : i32, 0 : i32], [6442 : i32, 1 : i32, 1 : i32, 0 : i32]]}
      : (!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) -> ()
    %7 = llvm.load %6 : !llvm.ptr -> vector<8xi16>
    llvm.return %7 : vector<8xi16>
  }
}

// CHECK: ![[DECO1]] = !{![[DECO2:.*]], ![[DECO3:.*]]}
// CHECK: ![[DECO2]] = !{i32 6442, i32 0, i32 1, i32 0}
// CHECK: ![[DECO3]] = !{i32 6442, i32 1, i32 1, i32 0}

// -----
module {
  llvm.func spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(!llvm.ptr<1> {llvm.nonnull, llvm.writeonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.readonly}) attributes {no_unwind, will_return}
  llvm.func @blockstore2d_cache_control(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: vector<8xi32>) {
    %0 = llvm.mlir.undef : vector<2xi32>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.insertelement %arg4, %0[%1 : i32] : vector<2xi32>
    %4 = llvm.insertelement %arg5, %3[%2 : i32] : vector<2xi32>
    %5 = llvm.mlir.constant(8 : i32) : i32
    %6 = llvm.alloca %5 x i32 : (i32) -> !llvm.ptr
    llvm.store %arg6, %6 : vector<8xi32>, !llvm.ptr
    // CHECK-LABEL: call spir_func void @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj
    // CHECK-SAME: !spirv.DecorationCacheControlINTEL ![[DECO1:.*]]
    llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(%arg0, %arg1, %arg2, %arg3, %4, %6)
      {function_type = !llvm.func<void (ptr<1>, i32, i32, i32, vector<2xi32>, ptr)>, linkage = #llvm.linkage<external>, no_unwind,
       sym_name = "_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj", visibility_ = 0 : i64, will_return,
       xevm.DecorationCacheControl = [[6443 : i32, 0 : i32, 2 : i32, 0 : i32], [6443 : i32, 1 : i32, 2 : i32, 0 : i32]]}
      : (!llvm.ptr<1> {llvm.nonnull, llvm.writeonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.readonly}) -> ()
    llvm.return
  }
}

// CHECK: ![[DECO1]] = !{![[DECO2:.*]], ![[DECO3:.*]]}
// CHECK: ![[DECO2]] = !{i32 6443, i32 0, i32 2, i32 0}
// CHECK: ![[DECO3]] = !{i32 6443, i32 1, i32 2, i32 0}

// -----
module {
  llvm.func spir_funccc @_Z44intel_sub_group_2d_block_prefetch_8b_8r32x1cPU3AS1viiiDv2_i(!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}
  llvm.func @blockprefetch2d(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
    %0 = llvm.mlir.undef : vector<2xi32>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.insertelement %arg4, %0[%1 : i32] : vector<2xi32>
    %4 = llvm.insertelement %arg5, %3[%2 : i32] : vector<2xi32>
    // CHECK-LABEL: call spir_func void @_Z44intel_sub_group_2d_block_prefetch_8b_8r32x1cPU3AS1viiiDv2_i
    // CHECK-SAME: !spirv.DecorationCacheControlINTEL ![[DECO1:.*]]
    llvm.call spir_funccc @_Z44intel_sub_group_2d_block_prefetch_8b_8r32x1cPU3AS1viiiDv2_i(%arg0, %arg1, %arg2, %arg3, %4)
      {function_type = !llvm.func<void (ptr<1>, i32, i32, i32, vector<2xi32>)>, linkage = #llvm.linkage<external>,
       memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind,
       sym_name = "_Z44intel_sub_group_2d_block_prefetch_8b_8r32x1cPU3AS1viiiDv2_i", visibility_ = 0 : i64,
       xevm.DecorationCacheControl = [[6442 : i32, 0 : i32, 1 : i32, 0 : i32], [6442 : i32, 1 : i32, 1 : i32, 0 : i32]]}
      : (!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) -> ()
    llvm.return
  }
}

// CHECK: ![[DECO1]] = !{![[DECO2:.*]], ![[DECO3:.*]]}
// CHECK: ![[DECO2]] = !{i32 6442, i32 0, i32 1, i32 0}
// CHECK: ![[DECO3]] = !{i32 6442, i32 1, i32 1, i32 0}

// -----
module {
  llvm.func spir_funccc @_Z8prefetchPU3AS1Kcm(!llvm.ptr<1>, i64) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}
  llvm.func @prefetch(%arg0: !llvm.ptr<1>) {
    %0 = llvm.mlir.constant(1 : i64) : i64
    // CHECK-LABEL: call spir_func void @_Z8prefetchPU3AS1Kcm
    // CHECK-SAME: !spirv.DecorationCacheControlINTEL ![[DECO1:.*]]
    llvm.call spir_funccc @_Z8prefetchPU3AS1Kcm(%arg0, %0)
      {function_type = !llvm.func<void (ptr<1>, i64)>, linkage = #llvm.linkage<external>,
       memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>,
       no_unwind, sym_name = "_Z8prefetchPU3AS1Kcm", visibility_ = 0 : i64,
       xevm.DecorationCacheControl = [[6442 : i32, 0 : i32, 1 : i32, 0 : i32], [6442 : i32, 1 : i32, 1 : i32, 0 : i32]]}
      : (!llvm.ptr<1>, i64) -> ()
    llvm.return
  }
}

// CHECK: ![[DECO1]] = !{![[DECO2:.*]], ![[DECO3:.*]]}
// CHECK: ![[DECO2]] = !{i32 6442, i32 0, i32 1, i32 0}
// CHECK: ![[DECO3]] = !{i32 6442, i32 1, i32 1, i32 0}

