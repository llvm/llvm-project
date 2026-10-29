; RUN: opt %s -passes='always-inline' -S | FileCheck %s

;; Every location in the first store's chain has its own layers: the store
;; (200), the location it is already inlined at (100) and the new call site
;; (300). Inlining @outer into @caller rebuilds all three, and each must keep
;; its own layers. The second store has none of its own and must not pick up
;; those of the location it is inlined at.

define void @outer(ptr %p, ptr %q) alwaysinline !dbg !6 {
  store ptr null, ptr %p, align 8, !dbg !9
  store ptr null, ptr %q, align 8, !dbg !24
  ret void, !dbg !8
}

define void @caller(ptr %p, ptr %q) !dbg !12 {
  call void @outer(ptr %p, ptr %q), !dbg !15
  ret void, !dbg !16
}

; CHECK-LABEL: define void @caller
; CHECK: store ptr null, ptr %p,{{.*}} !dbg ![[INST:[0-9]+]]
; CHECK-NEXT: store ptr null, ptr %q,{{.*}} !dbg ![[BARE:[0-9]+]]
; CHECK-DAG: ![[INST]] = distinct !DILocation(line: 6, column: 1, scope: ![[HELPER:[0-9]+]], inlinedAt: ![[OUTER:[0-9]+]], irlayers: ![[INST_LIST:[0-9]+]])
; CHECK-DAG: ![[BARE]] = distinct !DILocation(line: 7, column: 1, scope: ![[HELPER]], inlinedAt: ![[OUTER]])
; CHECK-DAG: ![[OUTER]] = distinct !DILocation(line: 11, column: 1, scope: ![[OUTER_SP:[0-9]+]], inlinedAt: ![[CS:[0-9]+]], irlayers: ![[OUTER_LIST:[0-9]+]])
; CHECK-DAG: ![[CS]] = distinct !DILocation(line: 15, column: 1, scope: ![[CALLER:[0-9]+]], irlayers: ![[CS_LIST:[0-9]+]])
; CHECK-DAG: ![[INST_LIST]] = !DILayerLocList(![[INST_LAYER:[0-9]+]])
; CHECK-DAG: ![[INST_LAYER]] = !DILayerLoc(line: 200, column: 1, file: {{![0-9]+}}, kind: "IntermediateIR")
; CHECK-DAG: ![[OUTER_LIST]] = !DILayerLocList(![[OUTER_LAYER:[0-9]+]])
; CHECK-DAG: ![[OUTER_LAYER]] = !DILayerLoc(line: 100, column: 1, file: {{![0-9]+}}, kind: "IntermediateIR")
; CHECK-DAG: ![[CS_LIST]] = !DILayerLocList(![[CS_LAYER:[0-9]+]])
; CHECK-DAG: ![[CS_LAYER]] = !DILayerLoc(line: 300, column: 1, file: {{![0-9]+}}, kind: "IntermediateIR")

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DISubprogram(name: "outer", scope: !1, file: !1, line: 10, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!7 = distinct !DISubprogram(name: "helper", scope: !1, file: !1, line: 5, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 10, column: 1, scope: !6)
!9 = !DILocation(line: 6, column: 1, scope: !7, inlinedAt: !10, irlayers: !21)
!10 = !DILocation(line: 11, column: 1, scope: !6, irlayers: !17)
!12 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 20, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!15 = !DILocation(line: 15, column: 1, scope: !12, irlayers: !23)
!16 = !DILocation(line: 16, column: 1, scope: !12)
!17 = !DILayerLocList(!19)
!18 = !DIFile(filename: "intermediate.ir", directory: "/")
!19 = !DILayerLoc(line: 100, column: 1, file: !18, kind: "IntermediateIR")
!20 = !DILayerLoc(line: 200, column: 1, file: !18, kind: "IntermediateIR")
!21 = !DILayerLocList(!20)
!22 = !DILayerLoc(line: 300, column: 1, file: !18, kind: "IntermediateIR")
!23 = !DILayerLocList(!22)
!24 = !DILocation(line: 7, column: 1, scope: !7, inlinedAt: !10)
