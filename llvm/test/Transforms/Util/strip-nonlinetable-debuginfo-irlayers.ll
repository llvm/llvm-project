; RUN: opt -passes=strip-nonlinetable-debuginfo -S %s | FileCheck %s

;; Layers are line-table data, so downgrading to line-tables-only keeps them,
;; as DILayerLoc/DILayerLocList nodes rather than plain tuples. The pass
;; rebuilds locations on two routes. Instruction !dbg attachments and the
;; locations inside !llvm.loop metadata share one lambda (@f, @loop). An
;; inlinedAt location goes through the metadata mapper instead, which is also
;; where the layer nodes would be rebuilt as plain tuples (@inlined).

define void @f(ptr %p) !dbg !5 {
  store ptr null, ptr %p, align 8, !dbg !20
  ret void, !dbg !21
}

define void @loop(ptr %p, i1 %c) !dbg !6 {
entry:
  br label %body

body:
  store ptr null, ptr %p, align 8, !dbg !22
  br i1 %c, label %body, label %exit, !llvm.loop !30

exit:
  ret void
}

define void @inlined(ptr %p) !dbg !7 {
  store ptr null, ptr %p, align 8, !dbg !40
  ret void
}

; CHECK-LABEL: define void @f
; CHECK:       store ptr null, ptr %p, align 8, !dbg ![[DBG:[0-9]+]]
; CHECK-LABEL: define void @loop
; CHECK:       store ptr null, ptr %p, align 8, !dbg ![[LOOPDBG:[0-9]+]]
; CHECK:       br {{.*}}, !llvm.loop ![[LOOPMD:[0-9]+]]
; CHECK-LABEL: define void @inlined
; CHECK:       store ptr null, ptr %p, align 8, !dbg ![[INLDBG:[0-9]+]]

; CHECK-DAG:   ![[DBG]] = !DILocation(line: 2, column: 5, scope: !{{[0-9]+}}, irlayers: ![[LIST:[0-9]+]])
; CHECK-DAG:   ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK-DAG:   ![[LAYER]] = !DILayerLoc(line: 42, column: 5, file: ![[INTF:[0-9]+]], kind: "IntermediateIR")
; CHECK-DAG:   ![[INTF]] = !DIFile(filename: "intermediate.ir", directory: ".")
;; The loop's start location is the store's node; its end location is its own.
; CHECK-DAG:   ![[LOOPDBG]] = !DILocation(line: 5, column: 3, scope: !{{[0-9]+}}, irlayers: ![[LIST]])
; CHECK-DAG:   ![[LOOPMD]] = distinct !{![[LOOPMD]], ![[LOOPDBG]], ![[LOOPEND:[0-9]+]],
; CHECK-DAG:   ![[LOOPEND]] = !DILocation(line: 7, column: 1, scope: !{{[0-9]+}}, irlayers: ![[LIST]])
;; The callee and inlined-at locations each keep their own list.
; CHECK-DAG:   ![[INLDBG]] = !DILocation(line: 13, column: 3, scope: !{{[0-9]+}}, inlinedAt: ![[IA:[0-9]+]], irlayers: ![[LIST]])
; CHECK-DAG:   ![[IA]] = !DILocation(line: 9, column: 7, scope: !{{[0-9]+}}, irlayers: ![[LIST2:[0-9]+]])
; CHECK-DAG:   ![[LIST2]] = !DILayerLocList(![[LAYER2:[0-9]+]])
; CHECK-DAG:   ![[LAYER2]] = !DILayerLoc(line: 77, column: 1, file: ![[INTF]], kind: "IntermediateIR")

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/test")
!2 = !{i32 7, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !{null})
!5 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!6 = distinct !DISubprogram(name: "loop", scope: !1, file: !1, line: 4, type: !4, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "inlined", scope: !1, file: !1, line: 8, type: !4, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !0)
!8 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 12, type: !4, scopeLine: 12, spFlags: DISPFlagDefinition, unit: !0)

!10 = !DIFile(filename: "intermediate.ir", directory: ".")
!11 = !DILayerLoc(line: 42, column: 5, file: !10, kind: "IntermediateIR")
!12 = !DILayerLocList(!11)
!13 = !DILayerLoc(line: 77, column: 1, file: !10, kind: "IntermediateIR")
!14 = !DILayerLocList(!13)

!20 = !DILocation(line: 2, column: 5, scope: !5, irlayers: !12)
!21 = !DILocation(line: 3, column: 1, scope: !5)
;; Loop ID operands: self, start location, end location, then options.
!22 = !DILocation(line: 5, column: 3, scope: !6, irlayers: !12)
!23 = !DILocation(line: 7, column: 1, scope: !6, irlayers: !12)
!30 = distinct !{!30, !22, !23, !31}
!31 = !{!"llvm.loop.mustprogress"}
;; A callee location inlined at a caller location, each with its own layers.
!40 = !DILocation(line: 13, column: 3, scope: !8, inlinedAt: !41, irlayers: !12)
!41 = !DILocation(line: 9, column: 7, scope: !7, irlayers: !14)
