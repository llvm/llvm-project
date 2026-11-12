; RUN: rm -rf %t && mkdir -p %t
; RUN: llc -split-dwarf-file=foo.dwo  %s -filetype=obj -o %t/a.o
; RUN: llc -split-dwarf-file=bar.dwo  %s -filetype=obj -o %t/b.o
; RUN: llvm-dwarfdump -debug-info %t/a.o %t/b.o | FileCheck %s

; CHECK: {{.*}}a.o: file format elf64-x86-64
; CHECK: 0x00000000: Compile Unit: {{.*}}, DWO_id = [[HASH:0x[0-9a-f]*]]
; CHECK: {{.*}}b.o: file format elf64-x86-64
; CHECK-NOT: DWO_id = [[HASH]]

target triple = "x86_64-pc-linux"

; Function Attrs: noinline nounwind uwtable
define void @_Z1av() !dbg !9 {
entry:
  ret void, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.ident = !{!5}
!llvm.module.flags = !{!6, !7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 304107) (llvm/trunk 304109)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "a.cpp", directory: "/tmp")
!2 = !{}
!5 = !{!"clang version 5.0.0 (trunk 304107) (llvm/trunk 304109)"}
!6 = !{i32 2, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = distinct !DISubprogram(name: "a", linkageName: "_Z1av", scope: !1, file: !1, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DILocation(line: 2, column: 1, scope: !9)
