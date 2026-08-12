; RUN: opt %s -passes='always-inline' -S | FileCheck %s

;; The store's location has no layers of its own; they are on the location it
;; is inlined at. Inlining @outer into @caller appends the call site to the
;; chain, and the location in @outer must keep its layers.

define void @outer(ptr %p) alwaysinline !dbg !6 {
  store ptr null, ptr %p, align 8, !dbg !9
  ret void, !dbg !8
}

define void @caller(ptr %p) !dbg !12 {
  call void @outer(ptr %p), !dbg !15
  ret void, !dbg !16
}

; CHECK-LABEL: define void @caller
; CHECK: store ptr null, ptr %p,{{.*}} !dbg ![[INST:[0-9]+]]
; CHECK-DAG: ![[INST]] = distinct !DILocation(line: 6, column: 1, scope: ![[HELPER:[0-9]+]], inlinedAt: ![[OUTER:[0-9]+]])
; CHECK-DAG: ![[OUTER]] = distinct !DILocation(line: 11, column: 1, scope: ![[OUTER_SP:[0-9]+]], inlinedAt: ![[CS:[0-9]+]], irlayers: ![[LIST:[0-9]+]])
; CHECK-DAG: ![[CS]] = distinct !DILocation(line: 15, column: 1, scope: ![[CALLER:[0-9]+]])
; CHECK-DAG: ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK-DAG: ![[LAYER]] = !DILayerLoc(line: 100, column: 1, file: {{![0-9]+}}, kind: "IntermediateIR")

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
!9 = !DILocation(line: 6, column: 1, scope: !7, inlinedAt: !10)
!10 = !DILocation(line: 11, column: 1, scope: !6, irlayers: !17)
!12 = distinct !DISubprogram(name: "caller", scope: !1, file: !1, line: 20, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!15 = !DILocation(line: 15, column: 1, scope: !12)
!16 = !DILocation(line: 16, column: 1, scope: !12)
!17 = !DILayerLocList(!19)
!18 = !DIFile(filename: "intermediate.ir", directory: "/")
!19 = !DILayerLoc(line: 100, column: 1, file: !18, kind: "IntermediateIR")
