; RUN: opt -S -passes=jump-threading < %s | FileCheck %s

@a = global i32 0, align 4
; Test that the llvm.dbg.value calls in a threaded block are correctly updated to
; target the locals in their threaded block, and not the unthreaded one.
define void @test2(i32 %cond1, i32 %cond2) {
; CHECK: [[globalptr:@.*]] = global i32 0, align 4
; CHECK: bb.cond2:
; CHECK: call void @llvm.dbg.value(metadata ptr null, metadata ![[DBG1ptr:[0-9]+]], metadata !DIExpression()), !dbg ![[DBG2ptr:[0-9]+]]
; CHECK-NEXT: [[TOBOOL1:%.*]] = icmp eq i32 %cond2, 0, !dbg ![[DBGLOCtobool1:[0-9]+]]
; CHECK-NEXT: call void @llvm.dbg.value(metadata !DIArgList(ptr null, i1 [[TOBOOL1]], i1 [[TOBOOL1]]), metadata !{{[0-9]+}}, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_LLVM_arg, 2, DW_OP_plus)), !dbg !{{[0-9]+}}
; CHECK: bb.cond2.thread:
; CHECK-NEXT: call void @llvm.dbg.value(metadata ptr [[globalptr]], metadata ![[DBG1ptr]], metadata !DIExpression()), !dbg ![[DBG2ptr]]
; CHECK-NEXT: [[TOBOOL12:%.*]] = icmp eq i32 %cond2, 0, !dbg ![[DBGLOCtobool1]]
; CHECK-NEXT: call void @llvm.dbg.value(metadata !DIArgList(ptr [[globalptr]], i1 [[TOBOOL12]], i1 [[TOBOOL12]]), metadata !{{[0-9]+}}, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_LLVM_arg, 2, DW_OP_plus)), !dbg !{{[0-9]+}}
entry:
  %tobool = icmp eq i32 %cond1, 0, !dbg !15
  call void @llvm.dbg.value(metadata i1 %tobool, metadata !9, metadata !DIExpression()), !dbg !15
  br i1 %tobool, label %bb.cond2, label %bb.f1, !dbg !16

bb.f1:                                            ; preds = %entry
  call void @f1(), !dbg !17
  br label %bb.cond2, !dbg !18

bb.cond2:                                         ; preds = %bb.f1, %entry
  %ptr = phi ptr [ null, %bb.f1 ], [ @a, %entry ], !dbg !19
  call void @llvm.dbg.value(metadata ptr %ptr, metadata !11, metadata !DIExpression()), !dbg !19
  %tobool1 = icmp eq i32 %cond2, 0, !dbg !20
  call void @llvm.dbg.value(metadata !DIArgList(ptr %ptr, i1 %tobool1, i1 %tobool1), metadata !13, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_LLVM_arg, 2, DW_OP_plus)), !dbg !20
  br i1 %tobool1, label %bb.file, label %bb.f2, !dbg !21

bb.f2:                                            ; preds = %bb.cond2
  call void @f2(), !dbg !22
  br label %exit, !dbg !23

bb.file:                                          ; preds = %bb.cond2
  %cmp = icmp eq ptr %ptr, null, !dbg !24
  call void @llvm.dbg.value(metadata i1 %cmp, metadata !14, metadata !DIExpression()), !dbg !24
  br i1 %cmp, label %bb.f4, label %bb.f3, !dbg !25

bb.f3:                                            ; preds = %bb.file
  br label %exit, !dbg !26

bb.f4:                                            ; preds = %bb.file
  call void @f4(), !dbg !27
  br label %exit, !dbg !28

exit:                                             ; preds = %bb.f4, %bb.f3, %bb.f2
  ret void, !dbg !29
}

declare void @f1()

declare void @f2()

declare void @f3()

declare void @f4()

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "/home/ben/Documents/llvm-project/llvm/test/Transforms/JumpThreading/thread-debug-info.ll", directory: "/")
!2 = !{i32 15}
!3 = !{i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test2", linkageName: "test2", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9, !11, !13, !14}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !10)
!10 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "2", scope: !5, file: !1, line: 5, type: !12)
!12 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!13 = !DILocalVariable(name: "3", scope: !5, file: !1, line: 6, type: !10)
!14 = !DILocalVariable(name: "4", scope: !5, file: !1, line: 10, type: !10)
!15 = !DILocation(line: 1, column: 1, scope: !5)
!16 = !DILocation(line: 2, column: 1, scope: !5)
!17 = !DILocation(line: 3, column: 1, scope: !5)
!18 = !DILocation(line: 4, column: 1, scope: !5)
!19 = !DILocation(line: 5, column: 1, scope: !5)
!20 = !DILocation(line: 6, column: 1, scope: !5)
!21 = !DILocation(line: 7, column: 1, scope: !5)
!22 = !DILocation(line: 8, column: 1, scope: !5)
!23 = !DILocation(line: 9, column: 1, scope: !5)
!24 = !DILocation(line: 10, column: 1, scope: !5)
!25 = !DILocation(line: 11, column: 1, scope: !5)
!26 = !DILocation(line: 12, column: 1, scope: !5)
!27 = !DILocation(line: 13, column: 1, scope: !5)
!28 = !DILocation(line: 14, column: 1, scope: !5)
!29 = !DILocation(line: 15, column: 1, scope: !5)