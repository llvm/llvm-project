; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s
;
; AMDGPUTargetLowering::performStoreCombine inserts a bitcast to convert
; <1 x float> to i32 for the store.  That synthetic bitcast must carry the
; value's (load's) debug location, not the store's, so that when
; visitBITCAST later folds the bitcast into the load the resulting load
; retains the correct source location.

; CHECK-LABEL: test:
; CHECK:       .loc 1 2 0
; CHECK-NEXT:  flat_load_dword

define void @test(ptr %p, ptr %q) !dbg !4 {
  %v = load <1 x float>, ptr %p, !dbg !7
  store <1 x float> %v, ptr %q, !dbg !8
  ret void
}

!llvm.dbg.cu = !{!0}
!9 = !{null}
!10 = !DISubroutineType(types: !9)
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "t.c", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", file: !1, line: 1, spFlags: DISPFlagDefinition, type: !10, unit: !0)
!7 = !DILocation(line: 2, scope: !4)
!8 = !DILocation(line: 3, scope: !4)
