; RUN: opt -passes=memcpyopt -S %s -verify-memoryssa | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@C = external constant [0 x i8]

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1)
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1)

define void @test_constant(i64 %src_size, ptr %dst, i64 %dst_size, i8 %c) !dbg !5 {
; CHECK-LABEL: @test_constant(
; CHECK-NEXT:    [[TMP1:%.*]] = icmp ule i64 [[DST_SIZE:%.*]], [[SRC_SIZE:%.*]], !dbg [[DBG11:![0-9]+]]
; CHECK-NEXT:    [[TMP2:%.*]] = sub i64 [[DST_SIZE]], [[SRC_SIZE]], !dbg [[DBG11]]
; CHECK-NEXT:    [[TMP3:%.*]] = select i1 [[TMP1]], i64 0, i64 [[TMP2]], !dbg [[DBG11]]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[DST:%.*]], i64 [[SRC_SIZE]], !dbg [[DBG11]]
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 1 [[TMP4]], i8 [[C:%.*]], i64 [[TMP3]], i1 false), !dbg [[DBG11]]
; CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr [[DST]], ptr @C, i64 [[SRC_SIZE]], i1 false), !dbg [[DBG12:![0-9]+]]
; CHECK-NEXT:    ret void, !dbg [[DBG13:![0-9]+]]
;
  call void @llvm.memset.p0.i64(ptr %dst, i8 %c, i64 %dst_size, i1 false), !dbg !11
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr @C, i64 %src_size, i1 false), !dbg !12
  ret void, !dbg !13
}

; Validate that the memset is mapped to DILocation for the original memset.
; CHECK: [[DBG11]] = !DILocation(line: 1,
; CHECK: [[DBG12]] = !DILocation(line: 2,
; CHECK: [[DBG13]] = !DILocation(line: 3,

!llvm.dbg.cu = !{!0}
!llvm.debugify = !{!2, !3}
!llvm.module.flags = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "memset-memcpy-dbgloc.ll", directory: "/")
!2 = !{i32 3}
!3 = !{i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test_constant", linkageName: "test_constant", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !{!9}
!9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 3, type: !10)
!10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_unsigned)
!11 = !DILocation(line: 1, column: 1, scope: !5)
!12 = !DILocation(line: 2, column: 1, scope: !5)
!13 = !DILocation(line: 3, column: 1, scope: !5)
