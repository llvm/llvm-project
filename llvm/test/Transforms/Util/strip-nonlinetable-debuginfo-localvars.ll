; RUN: opt -S -strip-nonlinetable-debuginfo %s -o - | FileCheck %s
; CHECK: define void @f() !dbg ![[F:[0-9]+]]
define void @f() !dbg !4 {
entry:
  %i = alloca i32, align 4
  ; CHECK-NOT: llvm.dbg.declare
  call void @llvm.dbg.declare(metadata i32* %i, metadata !11, metadata !13), !dbg !14
  store i32 42, i32* %i, align 4, !dbg !14
  ret void, !dbg !15
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "LLVM", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "f.c", directory: "/")
!2 = !{}
; CHECK: ![[F]] = distinct !DISubprogram(name: "f"
; CHECK-NOT: retainedNodes:
; CHECK-NOT: distinct !DISubprogram(name: "f"
!4 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 2}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"PIC Level", i32 2}
!10 = !{!"LLVM"}
!11 = !DILocalVariable(name: "i", scope: !4, file: !1, line: 1, type: !12)
!12 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIExpression()
!14 = !DILocation(line: 1, column: 16, scope: !4)
!15 = !DILocation(line: 1, column: 24, scope: !4)
