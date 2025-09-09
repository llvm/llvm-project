// RUN: mlir-opt --convert-ripple-to-llvm %s 2>%t | FileCheck %s

func.func @get_real(%k :i32, %n : i32) -> i32 {
    %two = arith.constant 2 : i32
    %result = arith.muli %two, %k : i32
    return %result : i32
}

func.func @get_imag(%k :i32, %n : i32) -> i32 {
    %two = arith.constant 2 : i32
    %tmp = arith.muli %two, %k : i32
    %one = arith.constant 1 : i32
    %result = arith.addi %tmp, %one : i32
    return %result : i32
}

// CHECK-LABEL:     @unzip
// CHECK:               %[[REAL:.*]] = llvm.mlir.addressof @get_real : !llvm.ptr
// CHECK:               %[[IMAG:.*]] = llvm.mlir.addressof @get_imag : !llvm.ptr

// CHECK:               %[[TRUE_1:.*]] = arith.constant true
// CHECK:               %{{.*}} = llvm.call_intrinsic "llvm.ripple.ishuffle"(%{{.*}}, %{{.*}}, %[[TRUE_1]], %[[REAL]])

// CHECK:               %[[TRUE_2:.*]] = arith.constant true
// CHECK:               %{{.*}} = llvm.call_intrinsic "llvm.ripple.ishuffle"(%{{.*}}, %{{.*}}, %[[TRUE_2]], %[[IMAG]])

func.func @unzip(%arr : memref<2x32xi32>) {
    %peid = arith.constant 0 : i32

    %dim_0 = arith.constant 0 : i32
    %dim_1 = arith.constant 1 : i32
    %size_1 = arith.constant 2 : i32
    %size_2 = arith.constant 32 : i32

    %bs = ripple.setshape %peid [%size_1, %size_2 : i32, i32] : i32 -> !ptr.ptr<#ptr.generic_space>
    %v0 = ripple.index (%bs : !ptr.ptr<#ptr.generic_space>) [%dim_0 : i32] -> i32
    %v1 = ripple.index (%bs : !ptr.ptr<#ptr.generic_space>) [%dim_1 : i32] -> i32

    %one = arith.constant 1 : i32
    %v0_mirror = arith.subi %one, %v0 : i32

    %v0_index = index.castu %v0 : i32 to index
    %v1_index = index.castu %v1 : i32 to index
    %v0_mirror_index = index.castu %v0_mirror : i32 to index

    %pair_id_0 = memref.load %arr[ %v0_index, %v1_index ] : memref<2x32xi32>
    %pair_id_1 = memref.load %arr[ %v0_mirror_index, %v1_index ] : memref<2x32xi32>

    %real = func.constant @get_real : (i32, i32) -> i32
    %imag = func.constant @get_imag : (i32, i32) -> i32

    %real_val = ripple.shuffle.pair [ %pair_id_0 : i32, %pair_id_1 : i32, %real : (i32, i32) -> i32 ] -> i32
    %imag_val = ripple.shuffle.pair [ %pair_id_0 : i32, %pair_id_1 : i32, %imag : (i32, i32) -> i32 ] -> i32

    return
}
