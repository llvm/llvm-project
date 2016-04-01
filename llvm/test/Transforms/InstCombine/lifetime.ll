; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.lifetime.start(i64, i8* nocapture)
declare void @llvm.lifetime.end(i64, i8* nocapture)
declare void @foo(i8* nocapture, i8* nocapture)

define void @bar(i1 %flag) !dbg !4 {
entry:
; CHECK-LABEL: @bar(
; CHECK: %[[T:[^ ]+]] = getelementptr inbounds [1 x i8], [1 x i8]* %text
; CHECK: %[[B:[^ ]+]] = getelementptr inbounds [1 x i8], [1 x i8]* %buff
; CHECK: if:
; CHECK-NEXT: br label %bb2
; CHECK: bb2:
; CHECK-NEXT: br label %bb3
; CHECK: bb3:
; CHECK-NEXT: call void @llvm.dbg.declare
; CHECK-NEXT: br label %fin
; CHECK: call void @llvm.lifetime.start(i64 1, i8* %[[T]])
; CHECK-NEXT: call void @llvm.lifetime.start(i64 1, i8* %[[B]])
; CHECK-NEXT: call void @foo(i8* %[[B]], i8* %[[T]])
; CHECK-NEXT: call void @llvm.lifetime.end(i64 1, i8* %[[B]])
; CHECK-NEXT: call void @llvm.lifetime.end(i64 1, i8* %[[T]])
  %text = alloca [1 x i8], align 1
  %buff = alloca [1 x i8], align 1
  %0 = getelementptr inbounds [1 x i8], [1 x i8]* %text, i64 0, i64 0
  %1 = getelementptr inbounds [1 x i8], [1 x i8]* %buff, i64 0, i64 0
  br i1 %flag, label %if, label %else

if:
  call void @llvm.lifetime.start(i64 1, i8* %0)
  call void @llvm.lifetime.start(i64 1, i8* %1)
  call void @llvm.lifetime.end(i64 1, i8* %1)
  call void @llvm.lifetime.end(i64 1, i8* %0)
  br label %bb2

bb2:
  call void @llvm.lifetime.start(i64 1, i8* %0)
  call void @llvm.lifetime.start(i64 1, i8* %1)
  call void @llvm.lifetime.end(i64 1, i8* %0)
  call void @llvm.lifetime.end(i64 1, i8* %1)
  br label %bb3

bb3:
  call void @llvm.lifetime.start(i64 1, i8* %0)
  call void @llvm.dbg.declare(metadata [1 x i8]* %text, metadata !14, metadata !25), !dbg !26
  call void @llvm.lifetime.end(i64 1, i8* %0)
  br label %fin

else:
  call void @llvm.lifetime.start(i64 1, i8* %0)
  call void @llvm.lifetime.start(i64 1, i8* %1)
  call void @foo(i8* %1, i8* %0)
  call void @llvm.lifetime.end(i64 1, i8* %1)
  call void @llvm.lifetime.end(i64 1, i8* %0)
  br  label %fin

fin:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 248826) (llvm/trunk 248827)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, subprograms: !3)
!1 = !DIFile(filename: "test.cpp", directory: "/home/user")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "bar", linkageName: "bar", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, variables: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "bool", size: 8, align: 8, encoding: DW_ATE_boolean)
!8 = !{!9, !11, !12, !14, !21}
!9 = !DILocalVariable(name: "Size", arg: 1, scope: !4, file: !1, line: 2, type: !10)
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "flag", arg: 2, scope: !4, file: !1, line: 2, type: !7)
!12 = !DILocalVariable(name: "i", scope: !13, file: !1, line: 3, type: !10)
!13 = distinct !DILexicalBlock(scope: !4, file: !1, line: 3, column: 3)
!14 = !DILocalVariable(name: "text", scope: !15, file: !1, line: 4, type: !17)
!15 = distinct !DILexicalBlock(scope: !16, file: !1, line: 3, column: 30)
!16 = distinct !DILexicalBlock(scope: !13, file: !1, line: 3, column: 3)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 8, align: 8, elements: !19)
!18 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!19 = !{!20}
!20 = !DISubrange(count: 1)
!21 = !DILocalVariable(name: "buff", scope: !15, file: !1, line: 5, type: !17)
!22 = !{i32 2, !"Dwarf Version", i32 4}
!23 = !{i32 2, !"Debug Info Version", i32 3}
!24 = !{!"clang version 3.8.0 (trunk 248826) (llvm/trunk 248827)"}
!25 = !DIExpression()
!26 = !DILocation(line: 4, column: 10, scope: !15)
