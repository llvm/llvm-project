; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s
;;
;; Outlining rebuilds every DILocation in the extracted region. The outermost
;; location of each inlinedAt chain is rebuilt with the cold function's scope,
;; and the locations inlined at it are rebuilt to point at the result. Check the
;; rebuilt scopes, that every location keeps its own irlayers through both
;; rebuilds, and that a location without layers does not pick up those of the
;; location it is inlined at.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@foo.cold.1(
; CHECK-SAME: !dbg ![[COLDSP:[0-9]+]]

;; !20 is not inlined, so it is the outermost location of its own chain.
; CHECK: [[ADD:%.*]] = add i32 %{{.*}}, 1, !dbg ![[OWN:[0-9]+]]

;; !21 is inlined at !22: !22 gets the cold function's scope, and !21 is
;; rebuilt to point at the new !22. !21 has no layers and must not get !22's.
; CHECK: call void @sink(i32 [[ADD]]), !dbg ![[INL:[0-9]+]]

;; !25 is also inlined at !22, but carries layers of its own, which the rebuild
;; must keep.
; CHECK: call void @sink(i32 %{{.*}}), !dbg ![[INLOWN:[0-9]+]]

;; !20 and !22 now have the cold function as their scope, while !21 and !25 keep
;; @inline_me's. Each location keeps its own coordinate: 100 for !20, 200 for
;; !22 and 300 for !25. !21 comes back with no irlayers; the closing paren is
;; the assertion.
; CHECK-DAG: ![[INLINESP:[0-9]+]] = distinct !DISubprogram(name: "inline_me"
; CHECK-DAG: ![[OWN]] = !DILocation(line: 1, column: 1, scope: ![[COLDSP]], irlayers: ![[OWNLIST:[0-9]+]])
; CHECK-DAG: ![[OWNLIST]] = !DILayerLocList(![[OWNLAYER:[0-9]+]])
; CHECK-DAG: ![[OWNLAYER]] = !DILayerLoc(line: 100, column: 1, file: !{{[0-9]+}}, kind: "IntermediateIR")

; CHECK-DAG: ![[INL]] = !DILocation(line: 2, column: 2, scope: ![[INLINESP]], inlinedAt: ![[IA:[0-9]+]])

; CHECK-DAG: ![[INLOWN]] = !DILocation(line: 4, column: 4, scope: ![[INLINESP]], inlinedAt: ![[IA]], irlayers: ![[OWNINLLIST:[0-9]+]])
; CHECK-DAG: ![[OWNINLLIST]] = !DILayerLocList(![[OWNINLLAYER:[0-9]+]])
; CHECK-DAG: ![[OWNINLLAYER]] = !DILayerLoc(line: 300, column: 1, file: !{{[0-9]+}}, kind: "IntermediateIR")

; CHECK-DAG: ![[IA]] = !DILocation(line: 3, column: 3, scope: ![[COLDSP]], irlayers: ![[IALIST:[0-9]+]])
; CHECK-DAG: ![[IALIST]] = !DILayerLocList(![[IALAYER:[0-9]+]])
; CHECK-DAG: ![[IALAYER]] = !DILayerLoc(line: 200, column: 1, file: !{{[0-9]+}}, kind: "IntermediateIR")

define void @foo(i32 %arg1, i1 %c) !dbg !6 {
entry:
  br i1 %c, label %if.then, label %if.end

if.then:
  ret void

if.end:
  %add1 = add i32 %arg1, 1, !dbg !20
  call void @sink(i32 %add1), !dbg !21
  call void @sink(i32 %arg1), !dbg !25
  ret void
}

declare void @sink(i32) cold

define void @inline_me() !dbg !12 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!12 = distinct !DISubprogram(name: "inline_me", linkageName: "inline_me", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)

;; Three distinct coordinates in the same intermediate file, so a location that
;; ends up with another one's layers fails rather than passing quietly.
!14 = !DIFile(filename: "intermediate.ir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
!15 = !DILayerLoc(line: 100, column: 1, file: !14, kind: "IntermediateIR")
!16 = !DILayerLocList(!15)
!17 = !DILayerLoc(line: 200, column: 1, file: !14, kind: "IntermediateIR")
!18 = !DILayerLocList(!17)
!23 = !DILayerLoc(line: 300, column: 1, file: !14, kind: "IntermediateIR")
!24 = !DILayerLocList(!23)

!20 = !DILocation(line: 1, column: 1, scope: !6, irlayers: !16)
!21 = !DILocation(line: 2, column: 2, scope: !12, inlinedAt: !22)
!22 = !DILocation(line: 3, column: 3, scope: !6, irlayers: !18)
!25 = !DILocation(line: 4, column: 4, scope: !12, inlinedAt: !22, irlayers: !24)
