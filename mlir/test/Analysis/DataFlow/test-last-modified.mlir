// RUN: mlir-opt -test-last-modified %s 2>&1 | FileCheck %s

// CHECK-LABEL: test_tag: test_simple_mod
// CHECK: operand #0
// CHECK-NEXT: - a
// CHECK: operand #1
// CHECK-NEXT: - b
func.func @test_simple_mod(%arg0: memref<i32>, %arg1: memref<i32>) -> (memref<i32>, memref<i32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  memref.store %c0, %arg0[] {tag_name = "a"} : memref<i32>
  memref.store %c1, %arg1[] {tag_name = "b"} : memref<i32>
  return {tag = "test_simple_mod"} %arg0, %arg1 : memref<i32>, memref<i32>
}

// CHECK-LABEL: test_tag: test_simple_mod_overwrite_a
// CHECK: operand #1
// CHECK-NEXT: - a
// CHECK-LABEL: test_tag: test_simple_mod_overwrite_b
// CHECK: operand #0
// CHECK-NEXT: - b
func.func @test_simple_mod_overwrite(%arg0: memref<i32>) -> memref<i32> {
  %c0 = arith.constant 0 : i32
  memref.store %c0, %arg0[] {tag = "test_simple_mod_overwrite_a", tag_name = "a"} : memref<i32>
  %c1 = arith.constant 1 : i32
  memref.store %c1, %arg0[] {tag_name = "b"} : memref<i32>
  return {tag = "test_simple_mod_overwrite_b"} %arg0 : memref<i32>
}

// CHECK-LABEL: test_tag: test_mod_control_flow
// CHECK: operand #0
// CHECK-NEXT: - b
// CHECK-NEXT: - a
func.func @test_mod_control_flow(%cond: i1, %ptr: memref<i32>) -> memref<i32> {
  cf.cond_br %cond, ^a, ^b

^a:
  %c0 = arith.constant 0 : i32
  memref.store %c0, %ptr[] {tag_name = "a"} : memref<i32>
  cf.br ^c

^b:
  %c1 = arith.constant 1 : i32
  memref.store %c1, %ptr[] {tag_name = "b"} : memref<i32>
  cf.br ^c

^c:
  return {tag = "test_mod_control_flow"} %ptr : memref<i32>
}

// CHECK-LABEL: test_tag: test_mod_dead_branch
// CHECK: operand #0
// CHECK-NEXT: - a
func.func @test_mod_dead_branch(%arg: i32, %ptr: memref<i32>) -> memref<i32> {
  %0 = arith.subi %arg, %arg : i32
  %1 = arith.constant -1 : i32
  %2 = arith.cmpi sgt, %0, %1 : i32
  cf.cond_br %2, ^a, ^b

^a:
  %c0 = arith.constant 0 : i32
  memref.store %c0, %ptr[] {tag_name = "a"} : memref<i32>
  cf.br ^c

^b:
  %c1 = arith.constant 1 : i32
  memref.store %c1, %ptr[] {tag_name = "b"} : memref<i32>
  cf.br ^c

^c:
  return {tag = "test_mod_dead_branch"} %ptr : memref<i32>
}

// CHECK-LABEL: test_tag: test_mod_region_control_flow
// CHECK: operand #0
// CHECK-NEXT: then
// CHECK-NEXT: else
func.func @test_mod_region_control_flow(%cond: i1, %ptr: memref<i32>) -> memref<i32> {
  scf.if %cond {
    %c0 = arith.constant 0 : i32
    memref.store %c0, %ptr[] {tag_name = "then"}: memref<i32>
  } else {
    %c1 = arith.constant 1 : i32
    memref.store %c1, %ptr[] {tag_name = "else"} : memref<i32>
  }
  return {tag = "test_mod_region_control_flow"} %ptr : memref<i32>
}

// CHECK-LABEL: test_tag: test_mod_dead_region
// CHECK: operand #0
// CHECK-NEXT: else
func.func @test_mod_dead_region(%ptr: memref<i32>) -> memref<i32> {
  %false = arith.constant false
  scf.if %false {
    %c0 = arith.constant 0 : i32
    memref.store %c0, %ptr[] {tag_name = "then"}: memref<i32>
  } else {
    %c1 = arith.constant 1 : i32
    memref.store %c1, %ptr[] {tag_name = "else"} : memref<i32>
  }
  return {tag = "test_mod_dead_region"} %ptr : memref<i32>
}

// CHECK-LABEL: test_tag: unknown_memory_effects_a
// CHECK: operand #1
// CHECK-NEXT: - a
// CHECK-LABEL: test_tag: unknown_memory_effects_b
// CHECK: operand #0
// CHECK-NEXT: - <unknown>
func.func @unknown_memory_effects(%ptr: memref<i32>) -> memref<i32> {
  %c0 = arith.constant 0 : i32
  memref.store %c0, %ptr[] {tag = "unknown_memory_effects_a", tag_name = "a"} : memref<i32>
  "test.unknown_effects"() : () -> ()
  return {tag = "unknown_memory_effects_b"} %ptr : memref<i32>
}

// CHECK-LABEL: test_tag: store_with_a_region_before::before:
// CHECK:  operand #0
// CHECK:   - pre
// CHECK: test_tag: inside_region:
// CHECK:  operand #0
// CHECK:   - region
// CHECK: test_tag: after:
// CHECK:  operand #0
// CHECK:   - region
// CHECK: test_tag: return:
// CHECK:  operand #0
// CHECK:   - post
func.func @store_with_a_region_before(%arg0: memref<f32>) -> memref<f32> {
  %0 = arith.constant 0.0 : f32
  %1 = arith.constant 1.0 : f32
  memref.store %0, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "store_with_a_region_before::before"} : memref<f32>
  test.store_with_a_region %arg0 attributes { tag_name = "region", store_before_region = true } {
    memref.load %arg0[] {tag = "inside_region"} : memref<f32>
    test.store_with_a_region_terminator
  } : memref<f32>
  memref.load %arg0[] {tag = "after"} : memref<f32>
  memref.store %1, %arg0[] {tag_name = "post"} : memref<f32>
  return {tag = "return"} %arg0 : memref<f32>
}

// CHECK-LABEL: test_tag: store_with_a_region_after::before:
// CHECK:  operand #0
// CHECK:   - pre
// CHECK: test_tag: inside_region:
// CHECK:  operand #0
// CHECK:   - pre
// CHECK: test_tag: after:
// CHECK:  operand #0
// CHECK:   - region
// CHECK: test_tag: return:
// CHECK:  operand #0
// CHECK:   - post
func.func @store_with_a_region_after(%arg0: memref<f32>) -> memref<f32> {
  %0 = arith.constant 0.0 : f32
  %1 = arith.constant 1.0 : f32
  memref.store %0, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "store_with_a_region_after::before"} : memref<f32>
  test.store_with_a_region %arg0 attributes { tag_name = "region", store_before_region = false } {
    memref.load %arg0[] {tag = "inside_region"} : memref<f32>
    test.store_with_a_region_terminator
  } : memref<f32>
  memref.load %arg0[] {tag = "after"} : memref<f32>
  memref.store %1, %arg0[] {tag_name = "post"} : memref<f32>
  return {tag = "return"} %arg0 : memref<f32>
}

// CHECK-LABEL: test_tag: store_with_a_region_before_containing_a_store::before:
// CHECK:  operand #0
// CHECK:   - pre
// CHECK: test_tag: enter_region:
// CHECK:  operand #0
// CHECK:   - region
// CHECK: test_tag: exit_region:
// CHECK:  operand #0
// CHECK:   - inner
// CHECK: test_tag: after:
// CHECK:  operand #0
// CHECK:   - inner
// CHECK: test_tag: return:
// CHECK:  operand #0
// CHECK:   - post
func.func @store_with_a_region_before_containing_a_store(%arg0: memref<f32>) -> memref<f32> {
  %0 = arith.constant 0.0 : f32
  %1 = arith.constant 1.0 : f32
  memref.store %0, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "store_with_a_region_before_containing_a_store::before"} : memref<f32>
  test.store_with_a_region %arg0 attributes { tag_name = "region", store_before_region = true } {
    memref.load %arg0[] {tag = "enter_region"} : memref<f32>
    %2 = arith.constant 2.0 : f32
    memref.store %2, %arg0[] {tag_name = "inner"} : memref<f32>
    memref.load %arg0[] {tag = "exit_region"} : memref<f32>
    test.store_with_a_region_terminator
  } : memref<f32>
  memref.load %arg0[] {tag = "after"} : memref<f32>
  memref.store %1, %arg0[] {tag_name = "post"} : memref<f32>
  return {tag = "return"} %arg0 : memref<f32>
}

// CHECK-LABEL: test_tag: store_with_a_region_after_containing_a_store::before:
// CHECK:  operand #0
// CHECK:   - pre
// CHECK: test_tag: enter_region:
// CHECK:  operand #0
// CHECK:   - pre
// CHECK: test_tag: exit_region:
// CHECK:  operand #0
// CHECK:   - inner
// CHECK: test_tag: after:
// CHECK:  operand #0
// CHECK:   - region
// CHECK: test_tag: return:
// CHECK:  operand #0
// CHECK:   - post
func.func @store_with_a_region_after_containing_a_store(%arg0: memref<f32>) -> memref<f32> {
  %0 = arith.constant 0.0 : f32
  %1 = arith.constant 1.0 : f32
  memref.store %0, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "store_with_a_region_after_containing_a_store::before"} : memref<f32>
  test.store_with_a_region %arg0 attributes { tag_name = "region", store_before_region = false } {
    memref.load %arg0[] {tag = "enter_region"} : memref<f32>
    %2 = arith.constant 2.0 : f32
    memref.store %2, %arg0[] {tag_name = "inner"} : memref<f32>
    memref.load %arg0[] {tag = "exit_region"} : memref<f32>
    test.store_with_a_region_terminator
  } : memref<f32>
  memref.load %arg0[] {tag = "after"} : memref<f32>
  memref.store %1, %arg0[] {tag_name = "post"} : memref<f32>
  return {tag = "return"} %arg0 : memref<f32>
}
