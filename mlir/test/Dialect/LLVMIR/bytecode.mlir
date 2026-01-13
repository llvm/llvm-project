// RUN: mlir-opt -verify-roundtrip %s

#access_group = #llvm.access_group<id = distinct[0]<>>
#access_group1 = #llvm.access_group<id = distinct[1]<>>
#di_subprogram = #llvm.di_subprogram<recId = distinct[2]<>>
#loc1 = loc("test.f90":12:14)
#loc2 = loc("test":4:3)
#loc6 = loc(fused<#di_subprogram>[#loc1])
#loc7 = loc(fused<#di_subprogram>[#loc2])
#loop_annotation = #llvm.loop_annotation<disableNonforced = false, mustProgress = true, startLoc = #loc6, endLoc = #loc7, parallelAccesses = #access_group, #access_group1>
module {
  llvm.func @imp_fn() {
    llvm.return loc(#loc2)
  } loc(#loc8)
  llvm.func @loop_annotation_with_locs() {
    llvm.br ^bb1 {loop_annotation = #loop_annotation} loc(#loc4)
  ^bb1:  // pred: ^bb0
    llvm.return loc(#loc5)
  } loc(#loc3)
} loc(#loc)
#di_file = #llvm.di_file<"test.f90" in "">
#di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_program>
#loc = loc("test":0:0)
#loc3 = loc("test-path":36:3)
#loc4 = loc("test-path":37:5)
#loc5 = loc("test-path":39:5)
#di_compile_unit = #llvm.di_compile_unit<id = distinct[3]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, isOptimized = false, emissionKind = Full>
#di_compile_unit1 = #llvm.di_compile_unit<id = distinct[4]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, isOptimized = false, emissionKind = Full>
#di_compile_unit2 = #llvm.di_compile_unit<id = distinct[5]<>, sourceLanguage = DW_LANG_Fortran95, file = #di_file, isOptimized = false, emissionKind = Full>
#di_module = #llvm.di_module<file = #di_file, scope = #di_compile_unit1, name = "mod1">
#di_module1 = #llvm.di_module<file = #di_file, scope = #di_compile_unit2, name = "mod2">
#di_imported_entity = #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #di_subprogram, entity = #di_module, file = #di_file, line = 1>
#di_imported_entity1 = #llvm.di_imported_entity<tag = DW_TAG_imported_module, scope = #di_subprogram, entity = #di_module1, file = #di_file, line = 1>
#di_subprogram1 = #llvm.di_subprogram<recId = distinct[2]<>, id = distinct[6]<>, compileUnit = #di_compile_unit, scope = #di_file, name = "imp_fn", file = #di_file, subprogramFlags = Definition, type = #di_subroutine_type, retainedNodes = #di_imported_entity, #di_imported_entity1>
#loc8 = loc(fused<#di_subprogram1>[#loc1])
