; RUN: opt -S -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

;; foldBranchToCommonDest moves the folded branch's atom group to the
;; predecessor's branch by copying its whole location, so it does so only when
;; the layers match. @test_no_transfer: they differ, the branch keeps its own
;; location. @test_transfer: they match, the atom moves, and both rebuilt
;; locations, the branch's and the cloned bonus instruction's, keep the layers.

declare void @a()
declare void @b()

define void @test_no_transfer(i32 %x, i1 %c1) !dbg !4 {
entry:
  br i1 %c1, label %merge, label %bb, !dbg !13

bb:
  %cond = icmp ne i32 %x, 0
  br i1 %cond, label %merge, label %other, !dbg !14

merge:
  call void @a()
  ret void

other:
  call void @b()
  ret void
}

;; The calls keep the branch from being folded away, so its location can be
;; checked directly.
; CHECK-LABEL: define {{.*}}@test_no_transfer
; CHECK:       %or.cond = select {{.*}}, !dbg ![[PRED:[0-9]+]]
; CHECK-NEXT:  br i1 %or.cond, {{.*}}, !dbg ![[PRED]]

define void @test_transfer(i32 %x, i1 %c0) !dbg !20 {
entry:
  br i1 %c0, label %merge, label %bb, !dbg !21

bb:
  %cmp = icmp eq i32 %x, 0, !dbg !22
  br i1 %cmp, label %merge, label %other, !dbg !23

merge:
  call void @a()
  ret void

other:
  call void @b()
  ret void
}

;; %cmp is cloned into entry with a fresh atom group (2 becomes 3), and the
;; branch takes the folded branch's atom, remapped the same way.
; CHECK-LABEL: define {{.*}}@test_transfer
; CHECK:       {{%.*}} = icmp eq i32 %x, 0, !dbg ![[CMP:[0-9]+]]
; CHECK-NEXT:  %or.cond = select
; CHECK-NEXT:  br i1 %or.cond, {{.*}}, !dbg ![[BR:[0-9]+]]

; CHECK:       ![[PRED]] = !DILocation(line: 10, column: 5, scope: !{{[0-9]+}}, irlayers: ![[LIST:[0-9]+]])
; CHECK:       ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK:       ![[LAYER]] = !DILayerLoc(line: 100,
; CHECK-DAG:   ![[CMP]] = !DILocation({{.*}}atomGroup: 3, atomRank: 2, irlayers: ![[LIST]])
; CHECK-DAG:   ![[BR]] = !DILocation({{.*}}atomGroup: 3, atomRank: 1, irlayers: ![[LIST]])

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test_no_transfer", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, keyInstructions: true)
!8 = !DIFile(filename: "intermediate.ir", directory: "/tmp")
!9 = !DILayerLoc(line: 100, column: 1, file: !8, kind: "IntermediateIR")
!10 = !DILayerLocList(!9)
!11 = !DILayerLoc(line: 200, column: 1, file: !8, kind: "IntermediateIR")
!12 = !DILayerLocList(!11)
;; Same position on both branches, different layers; only bb's has an atom.
!13 = !DILocation(line: 10, column: 5, scope: !4, irlayers: !10)
!14 = !DILocation(line: 10, column: 5, scope: !4, irlayers: !12, atomGroup: 1, atomRank: 1)

!20 = distinct !DISubprogram(name: "test_transfer", scope: !1, file: !1, line: 18, type: !2, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !0, keyInstructions: true)
;; Same position and layers everywhere; %cmp and bb's branch carry atoms,
;; entry's branch does not.
!21 = !DILocation(line: 20, column: 5, scope: !20, irlayers: !10)
!22 = !DILocation(line: 20, column: 5, scope: !20, irlayers: !10, atomGroup: 2, atomRank: 2)
!23 = !DILocation(line: 20, column: 5, scope: !20, irlayers: !10, atomGroup: 2, atomRank: 1)
