; RUN: llc -mtriple=amdgcn < %s | FileCheck %s

; Verify that folding bitcast(load) -> load in DAGCombiner preserves the
; original load's debug location, not the bitcast's.

; CHECK-LABEL: test:
; CHECK:       .loc 1 2 0 prologue_end
; CHECK-NEXT:  buffer_load_dword

define void @test(ptr addrspace(1) %p, ptr addrspace(1) %q) !dbg !4 {
  %v = load <1 x float>, ptr addrspace(1) %p, !dbg !7
  store <1 x float> %v, ptr addrspace(1) %q, !dbg !8
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "t.py", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", file: !1, line: 1, spFlags: DISPFlagDefinition, unit: !0)
!7 = !DILocation(line: 2, scope: !4)
!8 = !DILocation(line: 3, scope: !4)
