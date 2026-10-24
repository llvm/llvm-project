; RUN: opt -S -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

;; foldBranchToCommonDest moves the folded branch's atom group to the
;; predecessor's branch by copying its whole location, so it does so only when
;; the layers match too. Here they differ, so the branch keeps its own location.

declare void @a()
declare void @b()

define void @test_no_transfer(i32 %x, i1 %c1) !dbg !4 {
entry:
  br i1 %c1, label %merge, label %bb, !dbg !13

bb:
  %cond = icmp ne i32 %x, 0
  br i1 %cond, label %merge, label %other, !dbg !14

merge:
  call void @a(), !dbg !15
  ret void, !dbg !15

other:
  call void @b(), !dbg !16
  ret void, !dbg !16
}

;; The calls keep the branch from being folded away, so its location can be
;; checked directly.
; CHECK-LABEL: define {{.*}}@test_no_transfer
; CHECK:       %or.cond = select {{.*}}, !dbg ![[PRED:[0-9]+]]
; CHECK-NEXT:  br i1 %or.cond, {{.*}}, !dbg ![[PRED]]
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
!8 = !DIFile(filename: "intermediate.ir", directory: "/tmp")
!9  = !DILayerLoc(line: 100, column: 1, file: !8, kind: "IntermediateIR")
!10 = !DILayerLocList(!9)
!11 = !DILayerLoc(line: 200, column: 1, file: !8, kind: "IntermediateIR")
!12 = !DILayerLocList(!11)
;; Same position on both branches, different layers; only bb's has an atom.
!13 = !DILocation(line: 10, column: 5, scope: !4, irlayers: !10)
!14 = !DILocation(line: 10, column: 5, scope: !4, irlayers: !12,
                  atomGroup: 1, atomRank: 1)
!15 = !DILocation(line: 12, column: 3, scope: !4)
!16 = !DILocation(line: 14, column: 3, scope: !4)
