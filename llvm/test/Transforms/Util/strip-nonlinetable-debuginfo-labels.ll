; RUN: opt -S -strip-nonlinetable-debuginfo %s -o - | FileCheck %s
; CHECK: define void @f()
define void @f() !dbg !4 {
entry:
; CHECK-NOT: llvm.dbg.label
  call void @llvm.dbg.label(metadata !12), !dbg !11
  ret void, !dbg !11
}

declare void @llvm.dbg.label(metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "LLVM", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "f.c", directory: "/")
!2 = !{}
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"LLVM"}
!11 = !DILocation(line: 1, column: 24, scope: !4)
; CHECK-NOT: DILabel
!12 = !DILabel(scope: !4, name: "entry", file: !1, line: 1)
