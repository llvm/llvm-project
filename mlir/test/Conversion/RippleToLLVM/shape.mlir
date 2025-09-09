// RUN: mlir-opt --convert-ripple-to-llvm %s 2>%t | FileCheck %s

// -----

// CHECK-LABEL:   func @test_peid() {
// CHECK:           %[[BLOCK_SIZE:.*]] = llvm.call_intrinsic "llvm.ripple.block.setshape"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
// CHECK:           %{{.*}} = llvm.call_intrinsic "llvm.ripple.block.getsize"(%[[BLOCK_SIZE]], %{{.*}}) : (!llvm.ptr, [[TYPE:.*]]) -> [[TYPE]]

// CHECK:           %[[VZERO:.*]] = llvm.call_intrinsic "llvm.ripple.broadcast"(%[[BLOCK_SIZE]]
// CHECK:           %{{.*}} = llvm.call_intrinsic "llvm.ripple.slice"(%[[VZERO]]

func.func @test_peid() {
    %peid = arith.constant 0 : i32
    %dim = arith.constant 1 : i32
    %size_1 = arith.constant 2 : i32
    %size_2 = arith.constant 128 : i32

    %bs = ripple.setshape %peid [%size_1, %size_2 : i32, i32] : i32 -> !ptr.ptr<#ptr.generic_space>
    %nv = ripple.getsize (%bs : !ptr.ptr<#ptr.generic_space>) [%dim : i32] -> i32

    %zero = arith.constant 0 : i64
    %vzero = ripple.broadcast (%bs : !ptr.ptr<#ptr.generic_space>) [ %zero , %dim : i32] -> i32

    %slice = arith.constant -1 : i64
    %vzero_half = ripple.slice [%vzero : i32, %slice, %zero] -> i32

    return
}
