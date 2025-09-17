// RUN: mlir-opt --convert-ripple-to-llvm %s 2>%t | FileCheck %s

// -----

func.func @transpose8x8(%k : i32, %n : i32) -> i32 {
    %eight = arith.constant 8 : i32
    %res1 = arith.remui %k, %eight : i32
    %res2 = arith.divui %k, %eight : i32
    %imm = arith.addi %res1, %res2 : i32
    %result = arith.addi %eight, %imm : i32
    return %result : i32
}

// CHECK-LABEL:   func @transpose_tile
// CHECK:           %[[FUNC:.*]] = llvm.mlir.addressof @transpose8x8 : !llvm.ptr
// CHECK:           llvm.call_intrinsic "llvm.ripple.shuffle"(%{{.*}}, %{{.*}}, %false, %[[FUNC]])
func.func @transpose_tile(%tile_addr : memref<256xi32>, %v : i32) -> i32 {
    %peid = arith.constant 0 : i32
    %dim = arith.constant 1 : i32
    %size_1 = arith.constant 2 : i32
    %size_2 = arith.constant 128 : i32

    %bs = ripple.setshape %peid [%size_1, %size_2 : i32, i32] : i32 -> !ptr.ptr<#ptr.generic_space>
    %v0 = ripple.index (%bs : !ptr.ptr<#ptr.generic_space>) [%dim : i32] -> i32
    %nv = ripple.getsize (%bs : !ptr.ptr<#ptr.generic_space>) [%dim : i32] -> i32

    %v0_index = index.castu %v0 : i32 to index
    %tile_elem = memref.load %tile_addr[ %v0_index ] : memref<256xi32>
    %transpose8x8 = func.constant @transpose8x8 : (i32, i32) -> i32
    %result = ripple.shuffle [ %tile_elem : i32, %transpose8x8 : (i32, i32) -> i32 ] -> i32
    return %result : i32
}
