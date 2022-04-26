; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda | FileCheck --check-prefix=CHECK-NODIR %s
; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -dwarf-directory=1 | FileCheck --check-prefix=CHECK-DIR %s

; CHECK-NODIR: .file   {{[0-9]+}} "/tmp/dbginfo/a/a.cpp"
;
; ptxas does not support .file directory syntax, but it can still be
; forced by -dwarf-directory=1
; CHECK-DIR:   .file   {{[0-9]+}} "/tmp/dbginfo/a" "a.cpp"

define void @_Z4funcv() !dbg !4 {
entry:
  ret void, !dbg !5
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "a.cpp", directory: "/tmp/dbginfo/a")
!2 = !{}
!4 = distinct !DISubprogram(name: "func", linkageName: "_Z4funcv", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !1, type: !6, retainedNodes: !2)
!5 = !DILocation(line: 2, scope: !4)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}

