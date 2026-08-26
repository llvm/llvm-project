; RUN: opt -S -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

;; foldBranchToCommonDest: the Key Instructions atom-group transfer from BB's
;; terminator to the predecessor terminator (PTI) is guarded by
;; isSameSourceLocationAndIRLayers.  When PTI and BB's terminator share the
;; same primary source position but carry DIFFERENT irlayers, the transfer must
;; be skipped and PTI keeps its own location (irlayers line 100) rather than
;; receiving BB's location wholesale (irlayers line 200, atomGroup: 1).
;;
;; Non-vacuity: with the guard reverted to isSameSourceLocation the two
;; locations would compare equal (primary position matches) and the transfer
;; would fire, giving %or.cond a location with atomGroup: 1 and irlayers
;; pointing to line 200.  The CHECK below then fails against that output.

define i32 @test_no_transfer(i32 %x, i1 %c1) !dbg !4 {
entry:
  br i1 %c1, label %merge, label %bb, !dbg !13

bb:
  %cond = icmp ne i32 %x, 0
  br i1 %cond, label %merge, label %other, !dbg !14

merge:
  ret i32 1

other:
  ret i32 0
}

;; The folded select must carry the pred location (irlayers line 100, no
;; atomGroup).  If the transfer had fired it would carry atomGroup: 1 and
;; irlayers pointing to line 200.
; CHECK-LABEL: define {{.*}}@test_no_transfer
; CHECK:       %or.cond = select {{.*}}, !dbg ![[PRED:[0-9]+]]
; CHECK:       ![[PRED]] = !DILocation(line: 10, column: 5, scope: !{{[0-9]+}}, irlayers: ![[LIST:[0-9]+]])
; CHECK:       ![[LIST]] = !DILayerLocList(![[LAYER:[0-9]+]])
; CHECK:       ![[LAYER]] = !DILayerLoc(line: 100,

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test_no_transfer", scope: !1, file: !1,
                             line: 1, type: !2, scopeLine: 1,
                             spFlags: DISPFlagDefinition, unit: !0,
                             keyInstructions: true)

!8 = !DIFile(filename: "test.tile.ir", directory: "/tmp",
             checksumkind: CSK_MD5,
             checksum: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

;; Two distinct intermediate layers at the same primary source position.  Using
;; different layer coordinates means the test can tell apart "kept pred's layers
;; (line 100)" from "got bb's layers (line 200)".
!9  = !DILayerLoc(line: 100, column: 1, file: !8, kind: "tile ir")
!10 = !DILayerLocList(!9)
!11 = !DILayerLoc(line: 200, column: 1, file: !8, kind: "tile ir")
!12 = !DILayerLocList(!11)

;; Pred terminator: same primary position as bb, different irlayers, no atomGroup.
!13 = !DILocation(line: 10, column: 5, scope: !4, irlayers: !10)
;; BB terminator: same primary position, different irlayers, has atomGroup.
!14 = !DILocation(line: 10, column: 5, scope: !4, irlayers: !12,
                  atomGroup: 1, atomRank: 1)
