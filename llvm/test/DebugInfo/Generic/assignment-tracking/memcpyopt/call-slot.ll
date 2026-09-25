; RUN: opt -passes=memcpyopt -S %s | FileCheck %s

; Verify that after call-slot forwarding collapses the %a -> %b -> %c
; memcpy chain, the surviving memcpy's !DIAssignID links to %c's assignment
; (the actual destination), not %a's. Before the fix, it incorrectly kept
; the !DIAssignID from the first hop, so %c's dbg_assign pointed at %a.
; Fix #213642

@constant = private constant [4 x i8] c"abc\00"

define i8 @memcpy_chain() !dbg !5 {
; CHECK-LABEL: define i8 @memcpy_chain()
; CHECK:       [[A:%.*]] = alloca [4 x i8], align 1
; CHECK-NEXT:    #dbg_assign(i1 poison, ![[VAR_A:[0-9]+]], !DIExpression(), ![[ID_ALLOCA_A:[0-9]+]], ptr [[A]], !DIExpression(), ![[LOC:[0-9]+]])
; CHECK-NEXT:  [[B:%.*]] = alloca [4 x i8], align 1
; CHECK-NEXT:    #dbg_assign(i1 poison, ![[VAR_B:[0-9]+]], !DIExpression(), ![[ID_ALLOCA_B:[0-9]+]], ptr [[B]], !DIExpression(), ![[LOC]])
; CHECK-NEXT:  [[C:%.*]] = alloca [4 x i8], align 1
; CHECK-NEXT:    #dbg_assign(i1 poison, ![[VAR_C:[0-9]+]], !DIExpression(), ![[ID_ALLOCA_C:[0-9]+]], ptr [[C]], !DIExpression(), ![[LOC]])
; CHECK-NEXT:  call void @llvm.memcpy.p0.p0.i64(ptr align 1 [[C]], ptr align 1 @constant, i64 4, i1 false), !DIAssignID ![[ID_C:[0-9]+]]
; CHECK-NEXT:    #dbg_assign(i1 poison, ![[VAR_A]], !DIExpression(), ![[ID_A:[0-9]+]], ptr [[A]], !DIExpression(), ![[LOC]])
; CHECK-NEXT:    #dbg_assign(i1 poison, ![[VAR_B]], !DIExpression(), ![[ID_B:[0-9]+]], ptr [[B]], !DIExpression(), ![[LOC]])
; CHECK-NEXT:    #dbg_assign(i1 poison, ![[VAR_C]], !DIExpression(), ![[ID_C]], ptr [[C]], !DIExpression(), ![[LOC]])
; CHECK-NEXT:  [[VALUE:%.*]] = load i8, ptr [[C]], align 1
; CHECK-NEXT:  ret i8 [[VALUE]]
  %a = alloca [4 x i8], align 1, !DIAssignID !10
    #dbg_assign(i1 poison, !11, !DIExpression(), !10, ptr %a, !DIExpression(), !9)
  %b = alloca [4 x i8], align 1, !DIAssignID !12
    #dbg_assign(i1 poison, !13, !DIExpression(), !12, ptr %b, !DIExpression(), !9)
  %c = alloca [4 x i8], align 1, !DIAssignID !14
    #dbg_assign(i1 poison, !15, !DIExpression(), !14, ptr %c, !DIExpression(), !9)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a, ptr align 1 @constant, i64 4, i1 false), !DIAssignID !16
    #dbg_assign(i1 poison, !11, !DIExpression(), !16, ptr %a, !DIExpression(), !9)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %b, ptr align 1 %a, i64 4, i1 false), !DIAssignID !17
    #dbg_assign(i1 poison, !13, !DIExpression(), !17, ptr %b, !DIExpression(), !9)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c, ptr align 1 %b, i64 4, i1 false), !DIAssignID !18
    #dbg_assign(i1 poison, !15, !DIExpression(), !18, ptr %c, !DIExpression(), !9)
  %value = load i8, ptr %c, align 1
  ret i8 %value
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "call-slot.c", directory: "/")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !8)
!5 = distinct !DISubprogram(name: "memcpy_chain", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!6 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 32, elements: !8)
!8 = !{}
!9 = !DILocation(line: 0, scope: !5)
!10 = distinct !DIAssignID()
!11 = !DILocalVariable(name: "a", scope: !5, file: !1, line: 2, type: !7)
!12 = distinct !DIAssignID()
!13 = !DILocalVariable(name: "b", scope: !5, file: !1, line: 2, type: !7)
!14 = distinct !DIAssignID()
!15 = !DILocalVariable(name: "c", scope: !5, file: !1, line: 2, type: !7)
!16 = distinct !DIAssignID()
!17 = distinct !DIAssignID()
!18 = distinct !DIAssignID()
