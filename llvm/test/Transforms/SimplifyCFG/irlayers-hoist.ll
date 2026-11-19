; RUN: opt -S -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

;; Hoisted instructions get a line 0 location in the function's scope that
;; keeps their layers.

declare void @a()
declare void @b()

;; FoldBranchToCommonDest clones %y, whose line differs from the branch's, into
;; entry.
define void @bonus(i32 %x, i1 %c1) !dbg !4 {
; CHECK-LABEL: define void @bonus(
; CHECK-SAME:  !dbg ![[BONUS_SP:[0-9]+]]
; CHECK:       %y = add i32 %x, 1, !dbg ![[BONUS_Y:[0-9]+]]
entry:
  br i1 %c1, label %merge, label %bb, !dbg !10

bb:
  %y = add i32 %x, 1, !dbg !11
  %cond = icmp ne i32 %y, 0, !dbg !10
  br i1 %cond, label %merge, label %other, !dbg !10

merge:
  call void @a(), !dbg !12
  ret void

other:
  call void @b(), !dbg !12
  ret void
}

;; As above, with %y's layers on the location it is inlined at.
define void @bonus_inlined(i32 %x, i1 %c1) !dbg !20 {
; CHECK-LABEL: define void @bonus_inlined(
; CHECK-SAME:  !dbg ![[INL_SP:[0-9]+]]
; CHECK:       %y = add i32 %x, 1, !dbg ![[INL_Y:[0-9]+]]
entry:
  br i1 %c1, label %merge, label %bb, !dbg !24

bb:
  %y = add i32 %x, 1, !dbg !25
  %cond = icmp ne i32 %y, 0, !dbg !24
  br i1 %cond, label %merge, label %other, !dbg !24

merge:
  call void @a(), !dbg !27
  ret void

other:
  call void @b(), !dbg !27
  ret void
}

;; speculativelyExecuteBB hoists %y out of %then.
define i32 @speculate(i32 %x, i1 %c) !dbg !30 {
; CHECK-LABEL: define i32 @speculate(
; CHECK-SAME:  !dbg ![[SPEC_SP:[0-9]+]]
; CHECK:       %y = add i32 %x, 1, !dbg ![[SPEC_Y:[0-9]+]]
entry:
  br i1 %c, label %then, label %end, !dbg !33

then:
  %y = add i32 %x, 1, !dbg !34
  br label %end

end:
  %r = phi i32 [ %y, %then ], [ %x, %entry ]
  ret i32 %r, !dbg !33
}

; CHECK:      ![[BONUS_Y]] = !DILocation(line: 0, scope: ![[BONUS_SP]], irlayers: ![[BONUS_LIST:[0-9]+]])
; CHECK-NEXT: ![[BONUS_LIST]] = !DILayerLocList(![[BONUS_LAYER:[0-9]+]])
; CHECK-NEXT: ![[BONUS_LAYER]] = !DILayerLoc(line: 150,
; CHECK:      ![[INL_Y]] = !DILocation(line: 0, scope: ![[INL_SP]], irlayers: ![[INL_LIST:[0-9]+]])
; CHECK-NEXT: ![[INL_LIST]] = !DILayerLocList(![[INL_LAYER:[0-9]+]])
; CHECK-NEXT: ![[INL_LAYER]] = !DILayerLoc(line: 250,
; CHECK:      ![[SPEC_Y]] = !DILocation(line: 0, scope: ![[SPEC_SP]], irlayers: ![[SPEC_LIST:[0-9]+]])
; CHECK-NEXT: ![[SPEC_LIST]] = !DILayerLocList(![[SPEC_LAYER:[0-9]+]])
; CHECK-NEXT: ![[SPEC_LAYER]] = !DILayerLoc(line: 350,

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/tmp")
!2 = !DISubroutineType(types: !{})
!3 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !DIFile(filename: "intermediate.ir", directory: "/tmp")

!4 = distinct !DISubprogram(name: "bonus", scope: !1, file: !1, line: 1, type: !2, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DILayerLoc(line: 150, column: 1, file: !5, kind: "IntermediateIR")
!7 = !DILayerLocList(!6)
!10 = !DILocation(line: 10, column: 5, scope: !4)
!11 = !DILocation(line: 11, column: 7, scope: !4, irlayers: !7)
!12 = !DILocation(line: 12, column: 3, scope: !4)

!20 = distinct !DISubprogram(name: "bonus_inlined", scope: !1, file: !1, line: 19, type: !2, scopeLine: 19, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!21 = distinct !DISubprogram(name: "callee", scope: !1, file: !1, line: 40, type: !2, scopeLine: 40, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!22 = !DILayerLoc(line: 250, column: 1, file: !5, kind: "IntermediateIR")
!23 = !DILayerLocList(!22)
!24 = !DILocation(line: 20, column: 5, scope: !20)
!25 = !DILocation(line: 41, column: 7, scope: !21, inlinedAt: !26)
!26 = !DILocation(line: 21, column: 7, scope: !20, irlayers: !23)
!27 = !DILocation(line: 22, column: 3, scope: !20)

!30 = distinct !DISubprogram(name: "speculate", scope: !1, file: !1, line: 29, type: !2, scopeLine: 29, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!31 = !DILayerLoc(line: 350, column: 1, file: !5, kind: "IntermediateIR")
!32 = !DILayerLocList(!31)
!33 = !DILocation(line: 30, column: 5, scope: !30)
!34 = !DILocation(line: 31, column: 7, scope: !30, irlayers: !32)
