; RUN: opt -S -passes=speculative-execution < %s | FileCheck %s

;; A hoisted instruction gets a line 0 location in the function's scope that
;; keeps its layers.

define void @ifThen(i32 %x, i1 %c) !dbg !4 {
; CHECK-LABEL: define void @ifThen(
; CHECK-SAME:  !dbg ![[SP:[0-9]+]]
; CHECK-NEXT:    %y = add i32 %x, 1, !dbg ![[Y:[0-9]+]]
; CHECK-NEXT:    br i1 %c,
  br i1 %c, label %a, label %b, !dbg !8

a:
  %y = add i32 %x, 1, !dbg !9
  br label %b

b:
  ret void
}

; CHECK:      ![[Y]] = !DILocation(line: 0, scope: ![[SP]], irlayers: ![[LIST:[0-9]+]])
; CHECK-NEXT: ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK-NEXT: ![[LAYER]] = !DILayerLoc(line: 100,

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "ifThen", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DIFile(filename: "intermediate.ir", directory: "/tmp")
!6 = !DILayerLoc(line: 100, column: 1, file: !5, kind: "IntermediateIR")
!7 = !DILayerLocList(!6)
!8 = !DILocation(line: 2, column: 3, scope: !4)
!9 = !DILocation(line: 3, column: 5, scope: !4, irlayers: !7)
