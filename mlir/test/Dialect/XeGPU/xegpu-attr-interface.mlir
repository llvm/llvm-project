// RUN: mlir-opt --test-xegpu-layout-interface --cse -split-input-file %s | FileCheck %s

gpu.module @test {
  gpu.func @slice_attr() -> vector<128xindex> {
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[DIVU:.*]] = index.divu %[[SGID]], %[[C8:.*]]
    // CHECK-DAG: %[[REMU:.*]] = index.remu %[[DIVU]], %[[C4:.*]]
    // CHECK-DAG: %[[MUL:.*]] = index.mul %[[REMU]], %[[C32:.*]]
    // CHECK-DAG: %[[MOD:.*]] = index.remu %[[MUL]], %[[C128:.*]]
    // CHECK-DAG: %[[BASE:.*]] = vector.step : vector<32xindex>
    // CHECK-DAG: %[[CAST:.*]] = vector.broadcast %[[MOD]] : index to vector<32xindex>
    // CHECK-DAG: %[[ADD:.*]] = arith.addi %[[BASE]], %[[CAST]] : vector<32xindex>
    %step = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32]>, dims = [1]>}: vector<128xindex>
    gpu.return %step : vector<128xindex>
  }

  gpu.func @nested_slice_attr() -> vector<128xindex> {
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[DIVU1:.*]] = index.divu %[[SGID]], %[[C1:.*]]
    // CHECK-DAG: %[[DIVU2:.*]] = index.divu %[[DIVU1]], %[[C8:.*]]
    // CHECK-DAG: %[[REMU:.*]] = index.remu %[[DIVU2]], %[[C4:.*]]
    // CHECK-DAG: %[[MUL:.*]] = index.mul %[[REMU]], %[[C32:.*]]
    // CHECK-DAG: %[[MOD:.*]] = index.remu %[[MUL]], %[[C128:.*]]
    // CHECK-DAG: %[[BASE:.*]] = vector.step : vector<32xindex>
    // CHECK-DAG: %[[CAST:.*]] = vector.broadcast %[[MOD]] : index to vector<32xindex>
    // CHECK-DAG: %[[ADD:.*]] = arith.addi %[[BASE]], %[[CAST]] : vector<32xindex>
    %0 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [32, 32, 1]>, dims = [2]>, dims = [1]>} : vector<128xindex>
    gpu.return %0 : vector<128xindex>
  }

}

