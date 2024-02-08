// RUN: mlir-opt -test-last-modified --split-input-file %s 2>&1 |\
// RUN:          FileCheck %s --check-prefixes=CHECK,IP,IP_ONLY
// RUN: mlir-opt -test-last-modified='assume-func-writes=true' \
// RUN:          --split-input-file %s 2>&1 |\
// RUN:          FileCheck %s --check-prefixes=CHECK,IP,IP_AW
// RUN: mlir-opt -test-last-modified='interprocedural=false' \
// RUN:          --split-input-file %s 2>&1 |\
// RUN:          FileCheck %s --check-prefixes=CHECK,LOCAL
// RUN: mlir-opt \
// RUN:    -test-last-modified='interprocedural=false assume-func-writes=true' \
// RUN:    --split-input-file %s 2>&1 |\
// RUN:    FileCheck %s --check-prefixes=CHECK,LC_AW

// Check prefixes are as follows:
// 'check': common for all runs;
// 'ip': interprocedural runs;
// 'ip_aw': interpocedural runs assuming calls to external functions write to
//          all arguments;
// 'ip_only': interprocedural runs not assuming calls writing;
// 'local': local (non-interprocedural) analysis not assuming calls writing;
// 'lc_aw': local analysis assuming external calls writing to all arguments.

// CHECK-LABEL: test_tag: test_callsite
// IP:    operand #0
// IP-NEXT: - a
// LOCAL: operand #0
// LOCAL-NEXT: - <unknown>
// LC_AW: operand #0
// LC_AW-NEXT: - <unknown>
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
// IP:    operand #0
// IP-NEXT:    - b
// LOCAL: operand #0
// LOCAL-NEXT: - <unknown>
// LC_AW: operand #0
// LC_AW-NEXT: - <unknown>
func.func private @single_return_site_fn(%ptr: memref<i32>) -> memref<i32> {
  %c0 = arith.constant 0 : i32
  memref.store %c0, %ptr[] {tag_name = "b"} : memref<i32>
  return %ptr : memref<i32>
}

// CHECK-LABEL: test_tag: test_multiple_callsites
// IP:    operand #0
// IP-NEXT:    write0
// IP-NEXT:    write1
// LOCAL: operand #0
// LOCAL-NEXT: - <unknown>
// LC_AW: operand #0
// LC_AW-NEXT: - <unknown>
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
// IP:    operand #0
// IP-NEXT:    return0
// IP-NEXT:    return1
// LOCAL: operand #0
// LOCAL-NEXT: - <unknown>
// LC_AW: operand #0
// LC_AW-NEXT: - <unknown>
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

// -----

// CHECK-LABEL: test_tag: after_call
// IP:    operand #0
// IP-NEXT:    - write0
// LOCAL: operand #0
// LOCAL-NEXT: - <unknown>
// LC_AW: operand #0
// LC_AW-NEXT: - func.call
func.func private @void_return(%ptr: memref<i32>) {
  return
}

func.func @test_call_void_return() {
  %ptr = memref.alloc() : memref<i32>
  %c0 = arith.constant 0 : i32
  memref.store %c0, %ptr[] {tag_name = "write0"} : memref<i32>
  func.call @void_return(%ptr) : (memref<i32>) -> ()
  memref.load %ptr[] {tag = "after_call"} : memref<i32>
  return
}

// -----

func.func private @callee(%arg0: memref<f32>) -> memref<f32> {
  %2 = arith.constant 2.0 : f32
  memref.load %arg0[] {tag = "call_and_store_before::enter_callee"} : memref<f32>
  memref.store %2, %arg0[] {tag_name = "callee"} : memref<f32>
  memref.load %arg0[] {tag = "exit_callee"} : memref<f32>
  return %arg0 : memref<f32>
}
// In this test, the "call" operation also stores to %arg0 itself before
// transferring control flow to the callee. Therefore, the order of accesses is
// "pre" -> "call" -> "callee" -> "post"

// CHECK-LABEL: test_tag: call_and_store_before::enter_callee:
// IP:     operand #0
// IP:      - call
// LOCAL:  operand #0
// LOCAL:   - <unknown>
// LC_AW:  operand #0
// LC_AW:   - <unknown>

// CHECK: test_tag: exit_callee:
// CHECK:  operand #0
// CHECK:   - callee

// CHECK: test_tag: before_call:
// CHECK:  operand #0
// CHECK:   - pre

// CHECK: test_tag: after_call:
// IP:     operand #0
// IP:      - callee
// LOCAL:  operand #0
// LOCAL:   - <unknown>
// LC_AW:  operand #0
// LC_AW:   - call

// CHECK: test_tag: return:
// CHECK:  operand #0
// CHECK:   - post
func.func @call_and_store_before(%arg0: memref<f32>) -> memref<f32> {
  %0 = arith.constant 0.0 : f32
  %1 = arith.constant 1.0 : f32
  memref.store %0, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "before_call"} : memref<f32>
  test.call_and_store @callee(%arg0), %arg0 {tag_name = "call", store_before_call = true} : (memref<f32>, memref<f32>) -> ()
  memref.load %arg0[] {tag = "after_call"} : memref<f32>
  memref.store %1, %arg0[] {tag_name = "post"} : memref<f32>
  return {tag = "return"} %arg0 : memref<f32>
}

// -----

func.func private @callee(%arg0: memref<f32>) -> memref<f32> {
  %2 = arith.constant 2.0 : f32
  memref.load %arg0[] {tag = "call_and_store_after::enter_callee"} : memref<f32>
  memref.store %2, %arg0[] {tag_name = "callee"} : memref<f32>
  memref.load %arg0[] {tag = "exit_callee"} : memref<f32>
  return %arg0 : memref<f32>
}

// In this test, the "call" operation also stores to %arg0 itself after getting
// control flow back from the callee. Therefore, the order of accesses is
// "pre" -> "callee" -> "call" -> "post"

// CHECK-LABEL: test_tag: call_and_store_after::enter_callee:
// IP:     operand #0
// IP:      - pre
// LOCAL:  operand #0
// LOCAL:   - <unknown>
// LC_AW:  operand #0
// LC_AW:   - <unknown>

// CHECK: test_tag: exit_callee:
// CHECK:  operand #0
// CHECK:   - callee

// CHECK: test_tag: before_call:
// CHECK:  operand #0
// CHECK:   - pre

// CHECK:    test_tag: after_call:
// IP:     operand #0
// IP:      - call
// LOCAL:  operand #0
// LOCAL:   - <unknown>
// LC_AW:  operand #0
// LC_AW:   - call

// CHECK: test_tag: return:
// CHECK:  operand #0
// CHECK:   - post
func.func @call_and_store_after(%arg0: memref<f32>) -> memref<f32> {
  %0 = arith.constant 0.0 : f32
  %1 = arith.constant 1.0 : f32
  memref.store %0, %arg0[] {tag_name = "pre"} : memref<f32>
  memref.load %arg0[] {tag = "before_call"} : memref<f32>
  test.call_and_store @callee(%arg0), %arg0 {tag_name = "call", store_before_call = false} : (memref<f32>, memref<f32>) -> ()
  memref.load %arg0[] {tag = "after_call"} : memref<f32>
  memref.store %1, %arg0[] {tag_name = "post"} : memref<f32>
  return {tag = "return"} %arg0 : memref<f32>
}

// -----

func.func private @void_return(%ptr: memref<i32>)

// CHECK-LABEL: test_tag: after_opaque_call:
// CHECK:        operand #0
// IP_ONLY:       - <unknown>
// IP_AW:         - func.call
func.func @test_opaque_call_return() {
  %ptr = memref.alloc() : memref<i32>
  %c0 = arith.constant 0 : i32
  memref.store %c0, %ptr[] {tag_name = "write0"} : memref<i32>
  func.call @void_return(%ptr) : (memref<i32>) -> ()
  memref.load %ptr[] {tag = "after_opaque_call"} : memref<i32>
  return
}
