; RUN: opt -passes=callsite-splitting -S < %s | FileCheck %s --implicit-check-not=dbg.value
; RUN: opt -passes=callsite-splitting -S < %s --try-experimental-debuginfo-iterators | FileCheck %s --implicit-check-not=dbg.value

;; Test that DebugLocs are preserved, and that dbg.values are duplicated.


; CHECK-LABEL: @test1
; CHECK:         #dbg_value(i32 0,
; CHECK-NEXT:    [[R1:%.+]] = call i32 @callee(i32 0, i32 %dd), !dbg [[DBG1:!.*]]
; CHECK:         #dbg_value(i32 0,
; CHECK-NEXT:    [[R2:%.+]] = call i32 @callee(i32 1, i32 %dd), !dbg [[DBG1]]
; CHECK-LABEL: CallSite:
; CHECK-NEXT:    phi i32 [ [[R2]], %land.rhs.split ], [ [[R1]], %entry.split ], !dbg [[DBG1]]
; CHECK-NEXT:    #dbg_value(i32 1,

declare void @llvm.dbg.value(metadata, metadata, metadata)

define i32 @test1(ptr dereferenceable(4) %cc, i32 %dd, i1 %arg) !dbg !6 {
entry:
  br i1 %arg, label %CallSite, label %land.rhs

land.rhs:                                         ; preds = %entry
  br label %CallSite

CallSite:                                         ; preds = %land.rhs, %entry
  %pv = phi i32 [ 0, %entry ], [ 1, %land.rhs ]
  call void @llvm.dbg.value(metadata i32 0, metadata !9, metadata !DIExpression()), !dbg !18
  %call = call i32 @callee(i32 %pv, i32 %dd), !dbg !18
  call void @llvm.dbg.value(metadata i32 1, metadata !9, metadata !DIExpression()), !dbg !18
  ret i32 %call
}

; CHECK-LABEL: @test2
; CHECK:         [[LV1:%.*]] = load i32, ptr %ptr, align 4, !dbg [[DBG_LV:!.*]]
; CHECK-NEXT:    #dbg_value(i32 0,
; CHECK-NEXT:    [[R1:%.+]] = call i32 @callee(i32 0, i32 10), !dbg [[DBG_CALL:!.*]]
; CHECK:         [[LV2:%.*]] = load i32, ptr %ptr, align 4, !dbg [[DBG_LV]]
; CHECK-NEXT:    #dbg_value(i32 0,
; CHECK-NEXT:    [[R2:%.+]] = call i32 @callee(i32 0, i32 %i), !dbg [[DBG_CALL]]
; CHECK-LABEL: CallSite:
; CHECK-NEXT:    phi i32 [ [[LV1]], %Header.split ], [ [[LV2]], %TBB.split ], !dbg [[DBG_LV]]
; CHECK-NEXT:    phi i32 [ [[R1]], %Header.split ], [ [[R2]], %TBB.split ], !dbg [[DBG_CALL]]
; CHECK-NEXT:    #dbg_value(i32 1,

define void @test2(ptr %ptr, i32 %i, i1 %arg) !dbg !19 {
Header:
  %tobool = icmp ne i32 %i, 10
  br i1 %tobool, label %TBB, label %CallSite

TBB:                                              ; preds = %Header
  br i1 %arg, label %CallSite, label %End

CallSite:                                         ; preds = %TBB, %Header
  %lv = load i32, ptr %ptr, align 4, !dbg !25
  call void @llvm.dbg.value(metadata i32 0, metadata !21, metadata !DIExpression()), !dbg !26
  %cv = call i32 @callee(i32 0, i32 %i), !dbg !26
  call void @llvm.dbg.value(metadata i32 1, metadata !21, metadata !DIExpression()), !dbg !26
  %sub = sub nsw i32 %lv, %cv
  br label %End

End:                                              ; preds = %CallSite, %TBB
  ret void
}

define i32 @callee(i32 %aa, i32 %bb) {
entry:
  %add = add nsw i32 %aa, %bb
  ret i32 %add
}

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!3, !4}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 23}
!4 = !{i32 11}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test", linkageName: "test", scope: null, file: !1, line: 3, type: !7, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9, !11, !13, !15, !16, !17}
!9 = !DILocalVariable(name: "2", scope: !6, file: !1, line: 3, type: !10)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "3", scope: !6, file: !1, line: 5, type: !12)
!12 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "4", scope: !6, file: !1, line: 6, type: !14)
!14 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!15 = !DILocalVariable(name: "5", scope: !6, file: !1, line: 9, type: !12)
!16 = !DILocalVariable(name: "6", scope: !6, file: !1, line: 10, type: !12)
!17 = !DILocalVariable(name: "7", scope: !6, file: !1, line: 11, type: !10)
!18 = !DILocation(line: 10, column: 1, scope: !6)
!19 = distinct !DISubprogram(name: "test_add_new_phi", linkageName: "test_add_new_phi", scope: null, file: !1, line: 14, type: !7, isLocal: false, isDefinition: true, scopeLine: 14, isOptimized: true, unit: !0, retainedNodes: !20)
!20 = !{!21, !22, !23, !24}
!21 = !DILocalVariable(name: "8", scope: !19, file: !1, line: 14, type: !14)
!22 = !DILocalVariable(name: "9", scope: !19, file: !1, line: 17, type: !10)
!23 = !DILocalVariable(name: "10", scope: !19, file: !1, line: 18, type: !12)
!24 = !DILocalVariable(name: "11", scope: !19, file: !1, line: 20, type: !12)
!25 = !DILocation(line: 18, column: 1, scope: !19)
!26 = !DILocation(line: 19, column: 1, scope: !19)
