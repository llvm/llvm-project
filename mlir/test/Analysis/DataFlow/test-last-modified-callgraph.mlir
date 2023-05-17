// RUN: mlir-opt -test-last-modified %s 2>&1 | FileCheck %s

// CHECK-LABEL: test_tag: test_callsite
// CHECK: operand #0
// CHECK-NEXT: - a
func.func private @single_callsite_fn(%ptr: memref<i32>) -> memref<i32> {
  return {tag = "test_callsite"} %ptr : memref<i32>
}

func.func @test_callsite() {
  %ptr = memref.alloc() : memref<i32>
  %c0 = arith.constant 0 : i32
  memref.store %c0, %ptr[] {tag_name = "a"} : memref<i32>
  %0 = func.call @single_callsite_fn(%ptr) : (memref<i32>) -> memref<i32>
  return
}

// CHECK-LABEL: test_tag: test_return_site
// CHECK: operand #0
// CHECK-NEXT: - b
func.func private @single_return_site_fn(%ptr: memref<i32>) -> memref<i32> {
  %c0 = arith.constant 0 : i32
  memref.store %c0, %ptr[] {tag_name = "b"} : memref<i32>
  return %ptr : memref<i32>
}

// CHECK-LABEL: test_tag: test_multiple_callsites
// CHECK: operand #0
// CHECK-NEXT: write0
// CHECK-NEXT: write1
func.func @test_return_site(%ptr: memref<i32>) -> memref<i32> {
  %0 = func.call @single_return_site_fn(%ptr) : (memref<i32>) -> memref<i32>
  return {tag = "test_return_site"} %0 : memref<i32>
}

func.func private @multiple_callsite_fn(%ptr: memref<i32>) -> memref<i32> {
  return {tag = "test_multiple_callsites"} %ptr : memref<i32>
}

func.func @test_multiple_callsites(%a: i32, %ptr: memref<i32>) -> memref<i32> {
  memref.store %a, %ptr[] {tag_name = "write0"} : memref<i32>
  %0 = func.call @multiple_callsite_fn(%ptr) : (memref<i32>) -> memref<i32>
  memref.store %a, %ptr[] {tag_name = "write1"} : memref<i32>
  %1 = func.call @multiple_callsite_fn(%ptr) : (memref<i32>) -> memref<i32>
  return %ptr : memref<i32>
}

// CHECK-LABEL: test_tag: test_multiple_return_sites
// CHECK: operand #0
// CHECK-NEXT: return0
// CHECK-NEXT: return1
func.func private @multiple_return_site_fn(%cond: i1, %a: i32, %ptr: memref<i32>) -> memref<i32> {
  cf.cond_br %cond, ^a, ^b

^a:
  memref.store %a, %ptr[] {tag_name = "return0"} : memref<i32>
  return %ptr : memref<i32>

^b:
  memref.store %a, %ptr[] {tag_name = "return1"} : memref<i32>
  return %ptr : memref<i32>
}

func.func @test_multiple_return_sites(%cond: i1, %a: i32, %ptr: memref<i32>) -> memref<i32> {
  %0 = func.call @multiple_return_site_fn(%cond, %a, %ptr) : (i1, i32, memref<i32>) -> memref<i32>
  return {tag = "test_multiple_return_sites"} %0 : memref<i32>
}