; RUN: llc < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; RUN: llc < %s -filetype=obj -regalloc=basic | llvm-dwarfdump -debug-dump=info -  | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; Check debug info for variable z_s
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_variable
; CHECK: DW_TAG_variable
; CHECK-NEXT:   DW_AT_location
; CHECK-NEXT:   DW_AT_name {{.*}} "z_s"
; CHECK-NEXT:   DW_AT_decl_file
; CHECK-NEXT:   DW_AT_decl_line
; CHECK-NEXT:   DW_AT_type{{.*}}{[[TYPE:.*]]}
; CHECK: [[TYPE]]:
; CHECK-NEXT: DW_AT_name {{.*}} "int"


@.str1 = private unnamed_addr constant [14 x i8] c"m=%u, z_s=%d\0A\00"
@str = internal constant [21 x i8] c"Failing test vector:\00"

define i64 @gcd(i64 %a, i64 %b) nounwind readnone optsize noinline ssp !dbg !0 {
entry:
  tail call void @llvm.dbg.value(metadata i64 %a, i64 0, metadata !10, metadata !DIExpression()), !dbg !18
  tail call void @llvm.dbg.value(metadata i64 %b, i64 0, metadata !11, metadata !DIExpression()), !dbg !19
  br label %while.body, !dbg !20

while.body:                                       ; preds = %while.body, %entry
  %b.addr.0 = phi i64 [ %b, %entry ], [ %rem, %while.body ]
  %a.addr.0 = phi i64 [ %a, %entry ], [ %b.addr.0, %while.body ]
  %rem = srem i64 %a.addr.0, %b.addr.0, !dbg !21
  %cmp = icmp eq i64 %rem, 0, !dbg !23
  br i1 %cmp, label %if.then, label %while.body, !dbg !23

if.then:                                          ; preds = %while.body
  tail call void @llvm.dbg.value(metadata i64 %rem, i64 0, metadata !12, metadata !DIExpression()), !dbg !21
  ret i64 %b.addr.0, !dbg !23
}

define i32 @main() nounwind optsize ssp !dbg !6 {
entry:
  %call = tail call i32 @rand() nounwind optsize, !dbg !24
  tail call void @llvm.dbg.value(metadata i32 %call, i64 0, metadata !14, metadata !DIExpression()), !dbg !24
  %cmp = icmp ugt i32 %call, 21, !dbg !25
  br i1 %cmp, label %cond.true, label %cond.end, !dbg !25

cond.true:                                        ; preds = %entry
  %call1 = tail call i32 @rand() nounwind optsize, !dbg !25
  br label %cond.end, !dbg !25

cond.end:                                         ; preds = %entry, %cond.true
  %cond = phi i32 [ %call1, %cond.true ], [ %call, %entry ], !dbg !25
  tail call void @llvm.dbg.value(metadata i32 %cond, i64 0, metadata !17, metadata !DIExpression()), !dbg !25
  %conv = sext i32 %cond to i64, !dbg !26
  %conv5 = zext i32 %call to i64, !dbg !26
  %call6 = tail call i64 @gcd(i64 %conv, i64 %conv5) optsize, !dbg !26
  %cmp7 = icmp eq i64 %call6, 0
  br i1 %cmp7, label %return, label %if.then, !dbg !26

if.then:                                          ; preds = %cond.end
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @str, i64 0, i64 0))
  %call12 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str1, i64 0, i64 0), i32 %call, i32 %cond) nounwind optsize, !dbg !26
  ret i32 1, !dbg !27

return:                                           ; preds = %cond.end
  ret i32 0, !dbg !27
}

declare i32 @rand() optsize

declare i32 @printf(i8* nocapture, ...) nounwind optsize

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

declare i32 @puts(i8* nocapture) nounwind

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!33}

!0 = distinct !DISubprogram(name: "gcd", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, file: !31, scope: !1, type: !3, variables: !29)
!1 = !DIFile(filename: "rem_small.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 2.9 (trunk 124117)", isOptimized: true, emissionKind: 1, file: !31, enums: !32, retainedTypes: !32, subprograms: !28, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "main", line: 25, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, file: !31, scope: !1, type: !7, variables: !30)
!7 = !DISubroutineType(types: !8)
!8 = !{!9}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DILocalVariable(name: "a", line: 5, arg: 1, scope: !0, file: !1, type: !5)
!11 = !DILocalVariable(name: "b", line: 5, arg: 2, scope: !0, file: !1, type: !5)
!12 = !DILocalVariable(name: "c", line: 6, scope: !13, file: !1, type: !5)
!13 = distinct !DILexicalBlock(line: 5, column: 52, file: !31, scope: !0)
!14 = !DILocalVariable(name: "m", line: 26, scope: !15, file: !1, type: !16)
!15 = distinct !DILexicalBlock(line: 25, column: 12, file: !31, scope: !6)
!16 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!17 = !DILocalVariable(name: "z_s", line: 27, scope: !15, file: !1, type: !9)
!18 = !DILocation(line: 5, column: 41, scope: !0)
!19 = !DILocation(line: 5, column: 49, scope: !0)
!20 = !DILocation(line: 7, column: 5, scope: !13)
!21 = !DILocation(line: 8, column: 9, scope: !22)
!22 = distinct !DILexicalBlock(line: 7, column: 14, file: !31, scope: !13)
!23 = !DILocation(line: 9, column: 9, scope: !22)
!24 = !DILocation(line: 26, column: 38, scope: !15)
!25 = !DILocation(line: 27, column: 38, scope: !15)
!26 = !DILocation(line: 28, column: 9, scope: !15)
!27 = !DILocation(line: 30, column: 1, scope: !15)
!28 = !{!0, !6}
!29 = !{!10, !11, !12}
!30 = !{!14, !17}
!31 = !DIFile(filename: "rem_small.c", directory: "/private/tmp")
!32 = !{}
!33 = !{i32 1, !"Debug Info Version", i32 3}
