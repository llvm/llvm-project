// RUN: mlir-opt %s -inline -split-input-file | FileCheck %s

#di_file = #llvm.di_file<"file.mlir" in "/">

// CHECK: #[[START_ORIGINAL:.*]] = loc({{.*}}:42
#loc1 = loc("test.mlir":42:4)
// CHECK: #[[END_ORIGINAL:.*]] = loc({{.*}}:52
#loc2 = loc("test.mlir":52:4)
#loc3 = loc("test.mlir":62:4)
// CHECK: #[[CALL_ORIGINAL:.*]] = loc({{.*}}:72
#loc4 = loc("test.mlir":72:4)

#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file, isOptimized = false, emissionKind = None>
// CHECK: #[[CALLEE_DI:.*]] = #llvm.di_subprogram<{{.*}}, name = "callee"
#di_subprogram_callee = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "callee", file = #di_file, subprogramFlags = Definition>

// CHECK: #[[CALLER_DI:.*]] = #llvm.di_subprogram<{{.*}}, name = "caller"
#di_subprogram_caller = #llvm.di_subprogram<compileUnit = #di_compile_unit, scope = #di_file, name = "caller", file = #di_file, subprogramFlags = Definition>

// CHECK: #[[START_FUSED_ORIGINAL:.*]] = loc(fused<#[[CALLEE_DI]]>[#[[START_ORIGINAL]]
#start_loc_fused = loc(fused<#di_subprogram_callee>[#loc1])
// CHECK: #[[END_FUSED_ORIGINAL:.*]] = loc(fused<#[[CALLEE_DI]]>[#[[END_ORIGINAL]]
#end_loc_fused= loc(fused<#di_subprogram_callee>[#loc2])
#caller_loc= loc(fused<#di_subprogram_caller>[#loc3])
// CHECK: #[[CALL_FUSED:.*]] = loc(fused<#[[CALLER_DI]]>[#[[CALL_ORIGINAL]]
#call_loc= loc(fused<#di_subprogram_caller>[#loc4])

#loopMD = #llvm.loop_annotation<
        startLoc = #start_loc_fused,
        endLoc = #end_loc_fused>

// CHECK: #[[START_CALLSITE_LOC:.*]] = loc(callsite(#[[START_FUSED_ORIGINAL]] at #[[CALL_FUSED]]
// CHECK: #[[END_CALLSITE_LOC:.*]] = loc(callsite(#[[END_FUSED_ORIGINAL]] at #[[CALL_FUSED]]
// CHECK: #[[START_FUSED_LOC:.*]] = loc(fused<#[[CALLER_DI]]>[#[[START_CALLSITE_LOC]]
// CHECK: #[[END_FUSED_LOC:.*]] = loc(fused<#[[CALLER_DI]]>[
// CHECK: #[[LOOP_ANNOT:.*]] = #llvm.loop_annotation<
// CHECK-SAME: startLoc = #[[START_FUSED_LOC]], endLoc = #[[END_FUSED_LOC]]>

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
