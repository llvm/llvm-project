// RUN: mlir-opt %s -mlir-disable-threading -pass-pipeline='builtin.module(test-module-pass,func.func(test-function-pass)),func.func(test-function-pass)' -pass-pipeline="func.func(cse,canonicalize)" -verify-each=false -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck %s
// RUN: mlir-opt %s -mlir-disable-threading -test-textual-pm-nested-pipeline -verify-each=false -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck %s --check-prefix=TEXTUAL_CHECK
// RUN: mlir-opt %s -mlir-disable-threading -pass-pipeline='builtin.module(test-module-pass),any(test-interface-pass),any(test-interface-pass),func.func(test-function-pass),any(canonicalize),func.func(cse)' -verify-each=false -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck %s --check-prefix=GENERIC_MERGE_CHECK
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(test-module-pass' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_1 %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(test-module-pass))' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_2 %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module()(' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_3 %s
// RUN: not mlir-opt %s -pass-pipeline=',' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_4 %s
// RUN: not mlir-opt %s -pass-pipeline='func.func(test-module-pass)' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_5 %s

// CHECK_ERROR_1: encountered unbalanced parentheses while parsing pipeline
// CHECK_ERROR_2: encountered extra closing ')' creating unbalanced parentheses while parsing pipeline
// CHECK_ERROR_3: expected ',' after parsing pipeline
// CHECK_ERROR_4: does not refer to a registered pass or pass pipeline
// CHECK_ERROR_5:  Can't add pass '{{.*}}TestModulePass' restricted to 'builtin.module' on a PassManager intended to run on 'func.func', did you intend to nest?

func.func @foo() {
  return
}

module {
  func.func @foo() {
    return
  }
}

// CHECK: Pipeline Collection : ['builtin.module', 'func.func']
// CHECK-NEXT:   'func.func' Pipeline
// CHECK-NEXT:     TestFunctionPass
// CHECK-NEXT:     CSE
// CHECK-NEXT:       DominanceInfo
// CHECK-NEXT:     Canonicalizer
// CHECK-NEXT:   'builtin.module' Pipeline
// CHECK-NEXT:     TestModulePass
// CHECK-NEXT:     'func.func' Pipeline
// CHECK-NEXT:       TestFunctionPass

// TEXTUAL_CHECK: Pipeline Collection : ['builtin.module', 'func.func']
// TEXTUAL_CHECK-NEXT:   'func.func' Pipeline
// TEXTUAL_CHECK-NEXT:     TestFunctionPass
// TEXTUAL_CHECK-NEXT:   'builtin.module' Pipeline
// TEXTUAL_CHECK-NEXT:     TestModulePass
// TEXTUAL_CHECK-NEXT:     'func.func' Pipeline
// TEXTUAL_CHECK-NEXT:       TestFunctionPass

// Check that generic pass pipelines are only merged when they aren't
// going to overlap with op-specific pipelines.
// GENERIC_MERGE_CHECK:      Pipeline Collection : ['builtin.module', 'any']
// GENERIC_MERGE_CHECK-NEXT:   'any' Pipeline
// GENERIC_MERGE_CHECK-NEXT:     (anonymous namespace)::TestInterfacePass
// GENERIC_MERGE_CHECK-NEXT:   'builtin.module' Pipeline
// GENERIC_MERGE_CHECK-NEXT:     (anonymous namespace)::TestModulePass
// GENERIC_MERGE_CHECK-NEXT: 'any' Pipeline
// GENERIC_MERGE_CHECK-NEXT:   (anonymous namespace)::TestInterfacePass
// GENERIC_MERGE_CHECK-NEXT: 'func.func' Pipeline
// GENERIC_MERGE_CHECK-NEXT:   (anonymous namespace)::TestFunctionPass
// GENERIC_MERGE_CHECK-NEXT: 'any' Pipeline
// GENERIC_MERGE_CHECK-NEXT:   Canonicalizer
// GENERIC_MERGE_CHECK-NEXT: 'func.func' Pipeline
// GENERIC_MERGE_CHECK-NEXT:   CSE
// GENERIC_MERGE_CHECK-NEXT:     (A) DominanceInfo
