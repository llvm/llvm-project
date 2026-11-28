; RUN: opt -S -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

;; Check that SimplifyCFG does not replace a branch's irlayers when it folds
;; another branch into it and moves that branch's Key Instructions atom group
;; over. @test_atom_not_remapped: the predecessor's branch takes the atom and
;; keeps its own layers. @test_atom_remapped: the atom group is also remapped,
;; and each rebuilt location keeps its own layers.

declare void @a()
declare void @b()

define void @test_atom_not_remapped(i32 %x, i1 %c1) !dbg !4 {
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

;; %cond has no location, so the atom group is not remapped: entry's branch
;; takes the folded branch's atom as it is and keeps its own layers (line 100).
; CHECK-LABEL: define {{.*}}@test_atom_not_remapped
; CHECK:       %or.cond = select {{.*}}, !dbg ![[PRED:[0-9]+]]
; CHECK-NEXT:  br i1 %or.cond, {{.*}}, !dbg ![[PREDBR:[0-9]+]]

define void @test_atom_remapped(i32 %x, i1 %c0) !dbg !20 {
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
;; branch takes the folded branch's atom, remapped the same way. Each rebuilt
;; location keeps its own layers: line 300 on the clone, line 100 on the branch.
;; The select has the branch's original location.

; CHECK-LABEL: define {{.*}}@test_atom_remapped
; CHECK:       {{%.*}} = icmp eq i32 %x, 0, !dbg ![[CMP:[0-9]+]]
; CHECK-NEXT:  %or.cond = select {{.*}}, !dbg ![[SEL:[0-9]+]]
; CHECK-NEXT:  br i1 %or.cond, {{.*}}, !dbg ![[BR:[0-9]+]]

; CHECK:       ![[PRED]] = !DILocation(line: 10, column: 5, scope: !{{[0-9]+}}, irlayers: ![[LIST:[0-9]+]])
; CHECK:       ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK:       ![[LAYER]] = !DILayerLoc(line: 100,
; CHECK:       ![[PREDBR]] = !DILocation(line: 10, column: 5, scope: !{{[0-9]+}}, atomGroup: 1, atomRank: 1, irlayers: ![[LIST]])
; CHECK:       ![[CMP]] = !DILocation({{.*}}atomGroup: 3, atomRank: 2, irlayers: ![[CMPLIST:[0-9]+]])
; CHECK:       ![[CMPLIST]] = !DILayerLocList(![[CMPLAYER:[0-9]+]])
; CHECK:       ![[CMPLAYER]] = !DILayerLoc(line: 300,
; CHECK:       ![[SEL]] = !DILocation(line: 20, column: 5, scope: !{{[0-9]+}}, irlayers: ![[LIST]])
; CHECK:       ![[BR]] = !DILocation({{.*}}atomGroup: 3, atomRank: 1, irlayers: ![[LIST]])

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test_atom_not_remapped", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, keyInstructions: true)
!8 = !DIFile(filename: "intermediate.ir", directory: "/tmp")
!9 = !DILayerLoc(line: 100, column: 1, file: !8, kind: "IntermediateIR")
!10 = !DILayerLocList(!9)
!11 = !DILayerLoc(line: 200, column: 1, file: !8, kind: "IntermediateIR")
!12 = !DILayerLocList(!11)
!15 = !DILayerLoc(line: 300, column: 1, file: !8, kind: "IntermediateIR")
!16 = !DILayerLocList(!15)

;; Same position on both branches, different layers; only bb's has an atom.

!13 = !DILocation(line: 10, column: 5, scope: !4, irlayers: !10)
!14 = !DILocation(line: 10, column: 5, scope: !4, irlayers: !12, atomGroup: 1, atomRank: 1)

!20 = distinct !DISubprogram(name: "test_atom_remapped", scope: !1, file: !1, line: 18, type: !2, scopeLine: 18, spFlags: DISPFlagDefinition, unit: !0, keyInstructions: true)

;; As above, and %cmp shares the atom of bb's branch and has layers of its own.

!21 = !DILocation(line: 20, column: 5, scope: !20, irlayers: !10)
!22 = !DILocation(line: 20, column: 5, scope: !20, irlayers: !16, atomGroup: 2, atomRank: 2)
!23 = !DILocation(line: 20, column: 5, scope: !20, irlayers: !12, atomGroup: 2, atomRank: 1)
