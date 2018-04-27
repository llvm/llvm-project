; RUN: opt < %s -basicaa -loop-interchange -pass-remarks='loop-interchange' -pass-remarks-output=%t -S \
; RUN:     -verify-dom-info -verify-loop-info | FileCheck %s
; RUN: FileCheck -check-prefix=REMARK --input-file=%t %s


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i64]] zeroinitializer

;;  for(int i=0;i<100;i++)
;;    for(int j=0;j<100;j++)
;;      A[j][i] = A[j][i]+k;

; REMARK:      Name:            Interchanged
; REMARK-NEXT: Function:        interchange_01
; CHECK: split

define void @interchange_01(i64 %k, i64 %N) !dbg !5 {
entry:
  br label %for1.header

for1.header:
  %j23 = phi i64 [ 0, %entry ], [ %j.next24, %for1.inc10 ]
  call void @llvm.dbg.value(metadata i64 %j, metadata !13, metadata !DIExpression()), !dbg !14
  br label %for2

for2:
  %j = phi i64 [ %j.next, %for2 ], [ 0, %for1.header ]
  call void @llvm.dbg.value(metadata i64 %j, metadata !13, metadata !DIExpression()), !dbg !14
  %arrayidx5 = getelementptr inbounds [100 x [100 x i64]], [100 x [100 x i64]]* @A, i64 0, i64 %j, i64 %j23
  %lv = load i64, i64* %arrayidx5
  %add = add nsw i64 %lv, %k
  store i64 %add, i64* %arrayidx5
  %j.next = add nuw nsw i64 %j, 1
  %exitcond = icmp eq i64 %j, 99
  call void @llvm.dbg.value(metadata i64 %j, metadata !13, metadata !DIExpression()), !dbg !14
  br i1 %exitcond, label %for1.inc10, label %for2

for1.inc10:
  %j.next24 = add nuw nsw i64 %j23, 1
  call void @llvm.dbg.value(metadata i64 %j, metadata !13, metadata !DIExpression()), !dbg !14
  %exitcond26 = icmp eq i64 %j23, 99
  br i1 %exitcond26, label %for.end12, label %for1.header

for.end12:
  ret void
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.c", directory: "/test")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !12)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !8, !11}
!8 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 32, align: 32)
!10 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!11 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "a", arg: 1, scope: !5, file: !1, line: 1, type: !8)
!14 = !DILocation(line: 1, column: 27, scope: !5)
