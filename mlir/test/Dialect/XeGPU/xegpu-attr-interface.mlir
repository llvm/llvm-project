// RUN: mlir-opt --test-xegpu-layout-interface --cse -split-input-file %s | FileCheck %s

//CHECk: #map = affine_map<()[s0] -> (s0 floordiv 8)>
gpu.module @test {
  gpu.func @slice_attr() -> vector<128xindex> {
    //CHECK: [[sgId:%.+]] = gpu.subgroup_id : index
    //CHECK: [[IDY:%.+]] = affine.apply #map()[[[sgId]]]
    //CHECK: [[c32:%.+]] = arith.constant 32 : index
    //CHECK: [[LOCALY:%.+]] = index.mul [[IDY]], [[c32]]
    //CHECK: [[c0:%.+]] = arith.constant 0 : index
    //CHECK: [[Y:%.+]] = arith.addi [[LOCALY]], [[c0]] : index
    //CHECK: [[c128:%.+]] = arith.constant 128 : index
    //CHECK: [[MODY:%.+]] = index.remu [[Y]], [[c128]]
    //CHECK: [[BASE:%.+]] = vector.step : vector<32xindex>
    //CHECK: [[CAST:%.+]] = vector.broadcast [[MODY]] : index to vector<32xindex>
    //CHECK: [[ADD:%.+]] = arith.addi [[BASE]], [[CAST]] : vector<32xindex>
    %step = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32]>, dims = [1]>}: vector<128xindex>
    gpu.return %step : vector<128xindex>
  }

  gpu.func @nested_slice_attr() -> vector<128xindex> {
    //CHECK: [[sgId:%.+]] = gpu.subgroup_id : index
    //CHECK: [[IDY:%.+]] = affine.apply #map()[[[sgId]]]
    //CHECK: [[c32:%.+]] = arith.constant 32 : index
    //CHECK: [[LOCALY:%.+]] = index.mul [[IDY]], [[c32]]
    //CHECK: [[c0:%.+]] = arith.constant 0 : index
    //CHECK: [[Y:%.+]] = arith.addi [[LOCALY]], [[c0]] : index
    //CHECK: [[c128:%.+]] = arith.constant 128 : index
    //CHECK: [[MODY:%.+]] = index.remu [[Y]], [[c128]]
    //CHECK: [[BASE:%.+]] = vector.step : vector<32xindex>
    //CHECK: [[CAST:%.+]] = vector.broadcast [[MODY]] : index to vector<32xindex>
    //CHECK: [[ADD:%.+]] = arith.addi [[BASE]], [[CAST]] : vector<32xindex>
    %0 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [32, 32, 1]>, dims = [2]>, dims = [1]>} : vector<128xindex>
    gpu.return %0 : vector<128xindex>
  }

}