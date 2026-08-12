; RUN: opt -passes=hotcoldsplit -hotcoldsplit-threshold=0 -S < %s | FileCheck %s
;;
;; Outlining rebuilds every DILocation in the extracted region so its scope points
;; at the new cold function: once for the terminal frame, and once per inlined-at
;; frame. Check that irlayers survive both rebuilds.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

; CHECK-LABEL: define {{.*}}@foo.cold.1

;; A location carrying layers of its own: the terminal frame is rebuilt against
;; the cold function's scope and keeps its layer list.
; CHECK: [[ADD:%.*]] = add i32 %{{.*}}, 1, !dbg ![[OWN:[0-9]+]]

;; An inlined location whose inlinedAt frame carries the layers: the chain
;; rebuild has to keep them on that frame.
; CHECK: call void @sink(i32 [[ADD]]), !dbg ![[INL:[0-9]+]]

;; Each frame keeps its own coordinate: 100 for the terminal frame, 200 for the
;; inlined-at frame.
; CHECK-DAG: ![[OWN]] = !DILocation(line: 1, column: 1, scope: !{{[0-9]+}}, irlayers: ![[OWNLIST:[0-9]+]])
; CHECK-DAG: ![[OWNLIST]] = !DILayerLocList(![[OWNLAYER:[0-9]+]])
; CHECK-DAG: ![[OWNLAYER]] = !DILayerLoc(line: 100, column: 1, file: !{{[0-9]+}}, kind: "tile ir")

; CHECK-DAG: ![[INL]] = !DILocation(line: 2, column: 2, scope: !{{[0-9]+}}, inlinedAt: ![[IA:[0-9]+]])
; CHECK-DAG: ![[IA]] = !DILocation(line: 3, column: 3, scope: !{{[0-9]+}}, irlayers: ![[IALIST:[0-9]+]])
; CHECK-DAG: ![[IALIST]] = !DILayerLocList(![[IALAYER:[0-9]+]])
; CHECK-DAG: ![[IALAYER]] = !DILayerLoc(line: 200, column: 1, file: !{{[0-9]+}}, kind: "tile ir")

define void @foo(i32 %arg1, i1 %c) !dbg !6 {
entry:
  br i1 %c, label %if.then, label %if.end

if.then:
  ret void

if.end:
  %add1 = add i32 %arg1, 1, !dbg !20
  call void @sink(i32 %add1), !dbg !21
  ret void
}

declare void @sink(i32) cold

define void @inline_me() !dbg !12 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "tile", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !2)
!12 = distinct !DISubprogram(name: "inline_me", linkageName: "inline_me", scope: null, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)

;; Two distinct coordinates in one intermediate-IR snapshot, so a frame that ends
;; up with the other frame's layers fails rather than passing quietly.
!14 = !DIFile(filename: "kernel.tileir", directory: ".", checksumkind: CSK_MD5, checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
!15 = !DILayerLoc(line: 100, column: 1, file: !14, kind: "tile ir")
!16 = !DILayerLocList(!15)
!17 = !DILayerLoc(line: 200, column: 1, file: !14, kind: "tile ir")
!18 = !DILayerLocList(!17)

;; !20 carries layers directly, exercising the terminal-frame rebuild. !21 is
;; inlined from @inline_me and its inlinedAt frame !22 carries its own, exercising
;; the chain rebuild.
!20 = !DILocation(line: 1, column: 1, scope: !6, irlayers: !16)
!21 = !DILocation(line: 2, column: 2, scope: !12, inlinedAt: !22)
!22 = !DILocation(line: 3, column: 3, scope: !6, irlayers: !18)
