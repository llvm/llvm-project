// RUN: mlir-opt --test-xegpu-rematerialize-conversion-sources -allow-unregistered-dialect -split-input-file %s | FileCheck %s

// Row and column slices require different subgroup distributions.
#row = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>
#col = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>

// Rematerialization preserves the original value for its other consumer.
// CHECK-LABEL: gpu.func @step_conversion
// CHECK:         %[[STEP0:.*]] = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>} : vector<32xindex>
// CHECK-NEXT:    %[[STEP1:.*]] = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>} : vector<32xindex>
// CHECK:         arith.muli %[[STEP1]], %{{.*}} {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>} : vector<32xindex>
// CHECK:         vector.broadcast %[[STEP0]]
// CHECK-NOT:     xegpu.convert_layout
gpu.module @test {
gpu.func @step_conversion() kernel {
  %cst32 = arith.constant {layout_result_0 = #col} dense<32> : vector<32xindex>
  %step = vector.step {layout_result_0 = #row} : vector<32xindex>
  %cvt = xegpu.convert_layout %step <{input_layout = #row, target_layout = #col}> : vector<32xindex>
  %col = arith.muli %cvt, %cst32 {layout_result_0 = #col} : vector<32xindex>
  %rowb = vector.broadcast %step {layout_result_0 = #xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>} : vector<32xindex> to vector<32x32xindex>
  "some_use"(%col, %rowb) : (vector<32xindex>, vector<32x32xindex>) -> ()
  gpu.return
}
}

// -----

#row = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>
#col = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>

// The cloned elementwise chain shares no vector operands with the original.
// CHECK-LABEL: gpu.func @elementwise_chain_conversion
// CHECK-DAG:     %[[CST1:.*]] = arith.constant {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>} dense<32> : vector<32xindex>
// CHECK-DAG:     %[[STEP1:.*]] = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>} : vector<32xindex>
// CHECK:         %[[MUL1:.*]] = arith.muli %[[STEP1]], %[[CST1]] {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>} : vector<32xindex>
// CHECK:         vector.shape_cast %[[MUL1]]
// CHECK-NOT:     xegpu.convert_layout
gpu.module @test {
gpu.func @elementwise_chain_conversion() kernel {
  %cst32 = arith.constant {layout_result_0 = #row} dense<32> : vector<32xindex>
  %step = vector.step {layout_result_0 = #row} : vector<32xindex>
  %scaled = arith.muli %step, %cst32 {layout_result_0 = #row} : vector<32xindex>
  %cvt = xegpu.convert_layout %scaled <{input_layout = #row, target_layout = #col}> : vector<32xindex>
  %col2d = vector.shape_cast %cvt {layout_result_0 = #xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>} : vector<32xindex> to vector<32x1xindex>
  %rowb = vector.broadcast %scaled {layout_result_0 = #xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>} : vector<32xindex> to vector<32x32xindex>
  "some_use"(%col2d, %rowb) : (vector<32x1xindex>, vector<32x32xindex>) -> ()
  gpu.return
}
}

// -----

#row = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>
#col = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>

// The cloned cast and its step operand must have matching layouts.
// CHECK-LABEL: gpu.func @cast_chain_conversion
// CHECK:         %[[STEP0:.*]] = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>} : vector<32xindex>
// CHECK-NEXT:    %[[STEP1:.*]] = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>} : vector<32xindex>
// CHECK-NEXT:    arith.index_castui %[[STEP0]] {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>} : vector<32xindex> to vector<32xi32>
// CHECK-NEXT:    arith.index_castui %[[STEP1]] {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>} : vector<32xindex> to vector<32xi32>
// CHECK-NOT:     xegpu.convert_layout
gpu.module @test {
gpu.func @cast_chain_conversion() kernel {
  %step = vector.step {layout_result_0 = #row} : vector<32xindex>
  %cast = arith.index_castui %step {layout_result_0 = #row} : vector<32xindex> to vector<32xi32>
  %cvt = xegpu.convert_layout %cast <{input_layout = #row, target_layout = #col}> : vector<32xi32>
  %col2d = vector.shape_cast %cvt {layout_result_0 = #xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>} : vector<32xi32> to vector<32x1xi32>
  %row2d = vector.shape_cast %cast {layout_result_0 = #xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>} : vector<32xi32> to vector<1x32xi32>
  "some_use"(%col2d, %row2d) : (vector<32x1xi32>, vector<1x32xi32>) -> ()
  gpu.return
}
}

// -----

#row = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>
#col = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>

// Shared producers are cloned only once.
// CHECK-LABEL: gpu.func @shared_operand_cloned_once
// CHECK-COUNT-2: vector.step
// CHECK-NOT:     vector.step
// CHECK-NOT:     xegpu.convert_layout
// CHECK:         gpu.return
gpu.module @test {
gpu.func @shared_operand_cloned_once() kernel {
  %step = vector.step {layout_result_0 = #row} : vector<32xindex>
  %sq = arith.muli %step, %step {layout_result_0 = #row} : vector<32xindex>
  %cvt = xegpu.convert_layout %sq <{input_layout = #row, target_layout = #col}> : vector<32xindex>
  "some_use"(%cvt) : (vector<32xindex>) -> ()
  gpu.return
}
}

// -----

#row = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>
#col = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>

// Scalar block arguments remain shared because they carry no layout.
// CHECK-LABEL: gpu.func @scalar_operand_shared
// CHECK-SAME:    (%[[N:.*]]: index)
// CHECK:         vector.create_mask %[[N]] {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>} : vector<32xi1>
// CHECK-NEXT:    %[[MASK1:.*]] = vector.create_mask %[[N]] {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>} : vector<32xi1>
// CHECK:         "some_use"(%[[MASK1]])
// CHECK-NOT:     xegpu.convert_layout
gpu.module @test {
gpu.func @scalar_operand_shared(%n: index) kernel {
  %mask = vector.create_mask %n {layout_result_0 = #row} : vector<32xi1>
  %cvt = xegpu.convert_layout %mask <{input_layout = #row, target_layout = #col}> : vector<32xi1>
  "some_use"(%cvt) : (vector<32xi1>) -> ()
  gpu.return
}
}

// -----

#row = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>
#col = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>

// All conversion users share one rematerialized value.
// CHECK-LABEL: gpu.func @conversion_with_multiple_uses
// CHECK:         %[[STEP1:.*]] = vector.step {layout_result_0 = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>} : vector<32xindex>
// CHECK:         "use_a"(%[[STEP1]])
// CHECK:         "use_b"(%[[STEP1]])
// CHECK-NOT:     xegpu.convert_layout
gpu.module @test {
gpu.func @conversion_with_multiple_uses() kernel {
  %step = vector.step {layout_result_0 = #row} : vector<32xindex>
  %cvt = xegpu.convert_layout %step <{input_layout = #row, target_layout = #col}> : vector<32xindex>
  "use_a"(%cvt) : (vector<32xindex>) -> ()
  "use_b"(%cvt) : (vector<32xindex>) -> ()
  gpu.return
}
}

// -----

#row = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>
#col = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>

// An ineligible producer prevents rematerialization of the chain.
// CHECK-LABEL: gpu.func @opaque_source_kept
// CHECK:         xegpu.convert_layout
gpu.module @test {
gpu.func @opaque_source_kept() kernel {
  %v = "some_op"() {layout_result_0 = #row} : () -> vector<32xindex>
  %step = vector.step {layout_result_0 = #row} : vector<32xindex>
  %sum = arith.addi %v, %step {layout_result_0 = #row} : vector<32xindex>
  %cvt = xegpu.convert_layout %sum <{input_layout = #row, target_layout = #col}> : vector<32xindex>
  "some_use"(%cvt) : (vector<32xindex>) -> ()
  gpu.return
}
}

// -----

#row = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>

// Shape-changing broadcasts cannot share one layout across the chain.
// CHECK-LABEL: gpu.func @shape_changing_source_kept
// CHECK:         xegpu.convert_layout
gpu.module @test {
gpu.func @shape_changing_source_kept() kernel {
  %step = vector.step {layout_result_0 = #row} : vector<32xindex>
  %b = vector.broadcast %step {layout_result_0 = #xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>} : vector<32xindex> to vector<32x32xindex>
  %cvt = xegpu.convert_layout %b <{input_layout = #xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, target_layout = #xegpu.layout<sg_layout = [1, 32], sg_data = [32, 1]>}> : vector<32x32xindex>
  "some_use"(%cvt) : (vector<32x32xindex>) -> ()
  gpu.return
}
}

// -----

#row = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 32]>, dims = [0]>
#col = #xegpu.slice<#xegpu.layout<sg_layout = [32, 1], sg_data = [1, 1]>, dims = [1]>

// A chain of 17 ops exceeds the duplication limit.
// CHECK-LABEL: gpu.func @chain_too_long_kept
// CHECK:         xegpu.convert_layout
gpu.module @test {
gpu.func @chain_too_long_kept() kernel {
  %s = vector.step {layout_result_0 = #row} : vector<32xindex>
  %a1 = arith.addi %s, %s {layout_result_0 = #row} : vector<32xindex>
  %a2 = arith.addi %a1, %a1 {layout_result_0 = #row} : vector<32xindex>
  %a3 = arith.addi %a2, %a2 {layout_result_0 = #row} : vector<32xindex>
  %a4 = arith.addi %a3, %a3 {layout_result_0 = #row} : vector<32xindex>
  %a5 = arith.addi %a4, %a4 {layout_result_0 = #row} : vector<32xindex>
  %a6 = arith.addi %a5, %a5 {layout_result_0 = #row} : vector<32xindex>
  %a7 = arith.addi %a6, %a6 {layout_result_0 = #row} : vector<32xindex>
  %a8 = arith.addi %a7, %a7 {layout_result_0 = #row} : vector<32xindex>
  %a9 = arith.addi %a8, %a8 {layout_result_0 = #row} : vector<32xindex>
  %a10 = arith.addi %a9, %a9 {layout_result_0 = #row} : vector<32xindex>
  %a11 = arith.addi %a10, %a10 {layout_result_0 = #row} : vector<32xindex>
  %a12 = arith.addi %a11, %a11 {layout_result_0 = #row} : vector<32xindex>
  %a13 = arith.addi %a12, %a12 {layout_result_0 = #row} : vector<32xindex>
  %a14 = arith.addi %a13, %a13 {layout_result_0 = #row} : vector<32xindex>
  %a15 = arith.addi %a14, %a14 {layout_result_0 = #row} : vector<32xindex>
  %a16 = arith.addi %a15, %a15 {layout_result_0 = #row} : vector<32xindex>
  %cvt = xegpu.convert_layout %a16 <{input_layout = #row, target_layout = #col}> : vector<32xindex>
  "some_use"(%cvt) : (vector<32xindex>) -> ()
  gpu.return
}
}
