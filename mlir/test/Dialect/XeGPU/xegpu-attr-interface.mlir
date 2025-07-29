// RUN: mlir-opt --test-xegpu-layout-interface --cse -split-input-file %s | FileCheck %s

#block = #xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32]>
#slice = #xegpu.slice<#block, dims=[1]>

//CHECk: #map = affine_map<()[s0] -> (s0 floordiv 8)>
gpu.module @test_1_1_assignment {
  gpu.func @create_nd_tdesc() -> vector<128xindex> {
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
    %step = vector.step {layout_result_0 = #slice}: vector<128xindex>
    gpu.return %step : vector<128xindex>
  }
}