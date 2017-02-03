; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s

; FIXME: Should be .byte 8 for pointer size

; CHECK-LABEL: {{^}}basic_debug_info:
; CHECK: .section	.debug_info
; CHECK: .byte	4                               ; Address Size (in bytes)
define void @basic_debug_info() #0 !dbg !4 {
entry:
  ret void, !dbg !9
}

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/basic-debug-info.cl", directory: "/Users/matt/src/llvm/build_debug")
!2 = !{}
!4 = distinct !DISubprogram(name: "basic_debug_info", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !DILocation(line: 4, column: 1, scope: !4)
