// RUN: mlir-opt %s -mlir-disable-threading -pass-pipeline='builtin.module(builtin.module(test-module-pass,func.func(test-function-pass)),func.func(test-function-pass),func.func(cse,canonicalize))' -verify-each=false -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck %s
// RUN: mlir-opt %s -mlir-disable-threading -test-textual-pm-nested-pipeline -verify-each=false -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck %s --check-prefix=TEXTUAL_CHECK
// RUN: mlir-opt %s -mlir-disable-threading -pass-pipeline='builtin.module(builtin.module(test-module-pass),any(test-interface-pass),any(test-interface-pass),func.func(test-function-pass),any(canonicalize),func.func(cse))' -verify-each=false -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck %s --check-prefix=GENERIC_MERGE_CHECK
// RUN: mlir-opt %s -mlir-disable-threading -pass-pipeline=' builtin.module ( builtin.module( func.func( test-function-pass, print-op-stats{ json=false } ) ) ) ' -verify-each=false -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck %s --check-prefix=PIPELINE_STR_WITH_SPACES_CHECK
// RUN: not mlir-opt %s -pass-pipeline='any(builtin.module(test-module-pass)' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_1 %s
// RUN: not mlir-opt %s -pass-pipeline='builtin.module(test-module-pass))' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_2 %s
// RUN: not mlir-opt %s -pass-pipeline='any(builtin.module()()' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_3 %s
// RUN: not mlir-opt %s -pass-pipeline='any(,)' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_4 %s
// RUN: not mlir-opt %s -pass-pipeline='func.func(test-module-pass)' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_5 %s

// CHECK_ERROR_1: encountered unbalanced parentheses while parsing pipeline
// CHECK_ERROR_2: encountered extra closing ')' creating unbalanced parentheses while parsing pipeline
// CHECK_ERROR_3: expected ',' after parsing pipeline
// CHECK_ERROR_4: does not refer to a registered pass or pass pipeline
// CHECK_ERROR_5:  Can't add pass '{{.*}}TestModulePass' restricted to 'builtin.module' on a PassManager intended to run on 'func.func', did you intend to nest?

// RUN: not mlir-opt %s -pass-pipeline='' -cse 2>&1 | FileCheck --check-prefix=CHECK_ERROR_6 %s
// CHECK_ERROR_6: '-pass-pipeline' option can't be used with individual pass options

// RUN: not mlir-opt %s -pass-pipeline='wrong-op()' 2>&1 | FileCheck --check-prefix=CHECK_ERROR_7 %s
// CHECK_ERROR_7: can't run 'wrong-op' pass manager on 'builtin.module' op

// RUN: mlir-opt %s -pass-pipeline='any(cse)' -dump-pass-pipeline 2>&1 | FileCheck %s -check-prefix=CHECK_ROUNDTRIP
// CHECK_ROUNDTRIP: any(cse)

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

// PIPELINE_STR_WITH_SPACES_CHECK:   'builtin.module' Pipeline
// PIPELINE_STR_WITH_SPACES_CHECK-NEXT:   'func.func' Pipeline
// PIPELINE_STR_WITH_SPACES_CHECK-NEXT:     TestFunctionPass
// PIPELINE_STR_WITH_SPACES_CHECK-NEXT:     PrintOpStats

// Check that generic pass pipelines are only merged when they aren't
// going to overlap with op-specific pipelines.
// GENERIC_MERGE_CHECK:      Pipeline Collection : ['builtin.module', 'any']
// GENERIC_MERGE_CHECK-NEXT:   'any' Pipeline
// GENERIC_MERGE_CHECK-NEXT:     TestInterfacePass
// GENERIC_MERGE_CHECK-NEXT:   'builtin.module' Pipeline
// GENERIC_MERGE_CHECK-NEXT:     TestModulePass
// GENERIC_MERGE_CHECK-NEXT: 'any' Pipeline
// GENERIC_MERGE_CHECK-NEXT:   TestInterfacePass
// GENERIC_MERGE_CHECK-NEXT: 'func.func' Pipeline
// GENERIC_MERGE_CHECK-NEXT:   TestFunctionPass
// GENERIC_MERGE_CHECK-NEXT: 'any' Pipeline
// GENERIC_MERGE_CHECK-NEXT:   Canonicalizer
// GENERIC_MERGE_CHECK-NEXT: 'func.func' Pipeline
// GENERIC_MERGE_CHECK-NEXT:   CSE
// GENERIC_MERGE_CHECK-NEXT:     (A) DominanceInfo
