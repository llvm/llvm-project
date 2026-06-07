// RUN: mlir-opt %s -inline -split-input-file | FileCheck %s

#di_file = #llvm.di_file<"test.mlir" in "/">

// CHECK-DAG: #[[START_ORIGINAL:.*]] = loc({{.*}}:42:4)
#loc1 = loc("test.mlir":42:4)
// CHECK-DAG: #[[END_ORIGINAL:.*]] = loc({{.*}}:52:4)
#loc2 = loc("test.mlir":52:4)
#loc3 = loc("test.mlir":62:4)
// CHECK-DAG: #[[CALL_ORIGINAL:.*]] = loc({{.*}}:72:4)
#loc4 = loc("test.mlir":72:4)

#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false, emissionKind = None>
// CHECK-DAG: #[[CALLEE_DI:.*]] = #llvm.di_subprogram<{{.*}}, name = "callee"
#di_subprogram_callee = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "callee", file = #di_file, subprogramFlags = Definition>

// CHECK-DAG: #[[CALLER_DI:.*]] = #llvm.di_subprogram<{{.*}}, name = "caller"
#di_subprogram_caller = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "caller", file = #di_file, subprogramFlags = Definition>

// CHECK-DAG: #[[START_DI:.*]] = #llvm.di_location<#[[START_ORIGINAL]] in #[[CALLEE_DI]]>
#start_loc = #llvm.di_location<#loc1 in #di_subprogram_callee>
// CHECK-DAG: #[[END_DI:.*]] = #llvm.di_location<#[[END_ORIGINAL]] in #[[CALLEE_DI]]>
#end_loc = #llvm.di_location<#loc2 in #di_subprogram_callee>
#caller_loc = #llvm.di_location<#loc3 in #di_subprogram_caller>
// CHECK-DAG: #[[CALL_DI:.*]] = #llvm.di_location<#[[CALL_ORIGINAL]] in #[[CALLER_DI]]>
#call_loc = #llvm.di_location<#loc4 in #di_subprogram_caller>

#loopMD = #llvm.loop_annotation<
        startLoc = #start_loc,
        endLoc = #end_loc>

// After inlining, startLoc/endLoc are wrapped in CallSiteLoc.
// CHECK-DAG: #[[START_CALLSITE:.*]] = loc(callsite(#[[START_DI]] at #[[CALL_DI]]))
// CHECK-DAG: #[[END_CALLSITE:.*]] = loc(callsite(#[[END_DI]] at #[[CALL_DI]]))
// CHECK: #[[LOOP_ANNOT:.*]] = #llvm.loop_annotation<startLoc = #[[START_CALLSITE]], endLoc = #[[END_CALLSITE]]>

llvm.func @cond() -> i1

llvm.func @callee() {
  llvm.br ^head
^head:
  %c = llvm.call @cond() : () -> i1
  llvm.cond_br %c, ^head, ^exit {loop_annotation = #loopMD}
^exit:
  llvm.return
}

// CHECK: @loop_annotation
llvm.func @loop_annotation() {
  // CHECK: llvm.cond_br
  // CHECK-SAME: {loop_annotation = #[[LOOP_ANNOT]]
  llvm.call @callee() : () -> () loc(#call_loc)
  llvm.return
} loc(#caller_loc)
