// RUN: mlir-opt --test-xegpu-layout-interface --cse -split-input-file %s | FileCheck %s

gpu.module @test {
  gpu.func @slice_attr() -> vector<128xindex> {
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[DIVU:.*]] = arith.divui %[[SGID]], %[[C8:.*]]
    // CHECK-DAG: %[[REMU:.*]] = arith.remui %[[DIVU]], %[[C4:.*]]
    // CHECK-DAG: %[[MUL:.*]] = arith.muli %[[REMU]], %[[C32:.*]]
    // CHECK-DAG: %[[MOD:.*]] = arith.remui %[[MUL]], %[[C128:.*]]
    // CHECK-DAG: %[[BASE:.*]] = vector.step : vector<32xindex>
    // CHECK-DAG: %[[CAST:.*]] = vector.broadcast %[[MOD]] : index to vector<32xindex>
    // CHECK-DAG: %[[ADD:.*]] = arith.addi %[[BASE]], %[[CAST]] : vector<32xindex>
    %step = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [4, 8], sg_data = [32, 32]>, dims = [1]>}: vector<128xindex>
    gpu.return %step : vector<128xindex>
  }

  gpu.func @nested_slice_attr() -> vector<128xindex> {
    // CHECK-DAG: %[[SGID:.*]] = gpu.subgroup_id : index
    // CHECK-DAG: %[[DIVU2:.*]] = arith.divui %[[SGID]], %[[C8:.*]]
    // CHECK-DAG: %[[REMU:.*]] = arith.remui %[[DIVU2]], %[[C4:.*]]
    // CHECK-DAG: %[[MUL:.*]] = arith.muli %[[REMU]], %[[C32:.*]]
    // CHECK-DAG: %[[MOD:.*]] = arith.remui %[[MUL]], %[[C128:.*]]
    // CHECK-DAG: %[[BASE:.*]] = vector.step : vector<32xindex>
    // CHECK-DAG: %[[CAST:.*]] = vector.broadcast %[[MOD]] : index to vector<32xindex>
    // CHECK-DAG: %[[ADD:.*]] = arith.addi %[[BASE]], %[[CAST]] : vector<32xindex>
    %0 = vector.step {layout_result_0 = #xegpu.slice<#xegpu.slice<#xegpu.layout<sg_layout = [4, 8, 1], sg_data = [32, 32, 1]>, dims = [2]>, dims = [1]>} : vector<128xindex>
    gpu.return %0 : vector<128xindex>
  }

}
