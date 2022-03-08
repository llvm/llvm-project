// RUN: mlir-opt %s -mlir-disable-threading=true -verify-each=true -pass-pipeline='func.func(cse,canonicalize,cse)' -mlir-timing -mlir-timing-display=list 2>&1 | FileCheck -check-prefix=LIST %s
// RUN: mlir-opt %s -mlir-disable-threading=true -verify-each=true -pass-pipeline='func.func(cse,canonicalize,cse)' -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck -check-prefix=PIPELINE %s
// RUN: mlir-opt %s -mlir-disable-threading=false -verify-each=true -pass-pipeline='func.func(cse,canonicalize,cse)' -mlir-timing -mlir-timing-display=list 2>&1 | FileCheck -check-prefix=MT_LIST %s
// RUN: mlir-opt %s -mlir-disable-threading=false -verify-each=true -pass-pipeline='func.func(cse,canonicalize,cse)' -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck -check-prefix=MT_PIPELINE %s
// RUN: mlir-opt %s -mlir-disable-threading=true -verify-each=false -test-pm-nested-pipeline -mlir-timing -mlir-timing-display=tree 2>&1 | FileCheck -check-prefix=NESTED_PIPELINE %s

// LIST: Execution time report
// LIST: Total Execution Time:
// LIST: Name
// LIST-DAG: Canonicalizer
// LIST-DAG: CSE
// LIST-DAG: DominanceInfo
// LIST: Total

// PIPELINE: Execution time report
// PIPELINE: Total Execution Time:
// PIPELINE: Name
// PIPELINE-NEXT: Parser
// PIPELINE-NEXT: 'func.func' Pipeline
// PIPELINE-NEXT:   CSE
// PIPELINE-NEXT:     (A) DominanceInfo
// PIPELINE-NEXT:   Canonicalizer
// PIPELINE-NEXT:   CSE
// PIPELINE-NEXT:     (A) DominanceInfo
// PIPELINE-NEXT: Output
// PIPELINE-NEXT: Rest
// PIPELINE-NEXT: Total

// MT_LIST: Execution time report
// MT_LIST: Total Execution Time:
// MT_LIST: Name
// MT_LIST-DAG: Canonicalizer
// MT_LIST-DAG: CSE
// MT_LIST-DAG: DominanceInfo
// MT_LIST: Total

// MT_PIPELINE: Execution time report
// MT_PIPELINE: Total Execution Time:
// MT_PIPELINE: Name
// MT_PIPELINE-NEXT: Parser
// MT_PIPELINE-NEXT: 'func.func' Pipeline
// MT_PIPELINE-NEXT:   CSE
// MT_PIPELINE-NEXT:     (A) DominanceInfo
// MT_PIPELINE-NEXT:   Canonicalizer
// MT_PIPELINE-NEXT:   CSE
// MT_PIPELINE-NEXT:     (A) DominanceInfo
// MT_PIPELINE-NEXT: Output
// MT_PIPELINE-NEXT: Rest
// MT_PIPELINE-NEXT: Total

// NESTED_PIPELINE: Execution time report
// NESTED_PIPELINE: Total Execution Time:
// NESTED_PIPELINE: Name
// NESTED_PIPELINE-NEXT: Parser
// NESTED_PIPELINE-NEXT: Pipeline Collection : ['builtin.module', 'func.func']
// NESTED_PIPELINE-NEXT:   'func.func' Pipeline
// NESTED_PIPELINE-NEXT:     TestFunctionPass
// NESTED_PIPELINE-NEXT:   'builtin.module' Pipeline
// NESTED_PIPELINE-NEXT:     TestModulePass
// NESTED_PIPELINE-NEXT:     'func.func' Pipeline
// NESTED_PIPELINE-NEXT:       TestFunctionPass
// NESTED_PIPELINE-NEXT: Output
// NESTED_PIPELINE-NEXT: Rest
// NESTED_PIPELINE-NEXT: Total

func @foo() {
  return
}

func @bar() {
  return
}

func @baz() {
  return
}

func @foobar() {
  return
}

module {
  func @baz() {
    return
  }

  func @foobar() {
    return
  }
}
