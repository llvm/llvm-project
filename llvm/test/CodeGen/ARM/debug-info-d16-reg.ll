; RUN: llc < %s | FileCheck %s
; Radar 9309221
; Test dwarf reg no for d16
;CHECK: DW_OP_regx
;CHECK-NEXT: 272

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

@.str = private unnamed_addr constant [11 x i8] c"%p %lf %c\0A\00", align 4
@.str1 = private unnamed_addr constant [6 x i8] c"point\00", align 4

define i32 @inlineprinter(i8* %ptr, double %val, i8 zeroext %c) nounwind optsize !dbg !9 {
entry:
  tail call void @llvm.dbg.value(metadata i8* %ptr, metadata !19, metadata !DIExpression()), !dbg !26
  tail call void @llvm.dbg.value(metadata double %val, metadata !20, metadata !DIExpression()), !dbg !26
  tail call void @llvm.dbg.value(metadata i8 %c, metadata !21, metadata !DIExpression()), !dbg !26
  %0 = zext i8 %c to i32, !dbg !27
  %1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %val, i32 %0) nounwind, !dbg !27
  ret i32 0, !dbg !29
}

define i32 @printer(i8* %ptr, double %val, i8 zeroext %c) nounwind optsize noinline !dbg !0 {
entry:
  tail call void @llvm.dbg.value(metadata i8* %ptr, metadata !16, metadata !DIExpression()), !dbg !30
  tail call void @llvm.dbg.value(metadata double %val, metadata !17, metadata !DIExpression()), !dbg !30
  tail call void @llvm.dbg.value(metadata i8 %c, metadata !18, metadata !DIExpression()), !dbg !30
  %0 = zext i8 %c to i32, !dbg !31
  %1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %val, i32 %0) nounwind, !dbg !31
  ret i32 0, !dbg !33
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind optsize !dbg !10 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %argc, metadata !22, metadata !DIExpression()), !dbg !34
  tail call void @llvm.dbg.value(metadata i8** %argv, metadata !23, metadata !DIExpression()), !dbg !34
  %0 = sitofp i32 %argc to double, !dbg !35
  %1 = fadd double %0, 5.555552e+05, !dbg !35
  tail call void @llvm.dbg.value(metadata double %1, metadata !24, metadata !DIExpression()), !dbg !35
  %2 = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str1, i32 0, i32 0)) nounwind, !dbg !36
  %3 = getelementptr inbounds i8, i8* bitcast (i32 (i32, i8**)* @main to i8*), i32 %argc, !dbg !37
  %4 = trunc i32 %argc to i8, !dbg !37
  %5 = add i8 %4, 97, !dbg !37
  tail call void @llvm.dbg.value(metadata i8* %3, metadata !49, metadata !DIExpression()) nounwind, !dbg !38
  tail call void @llvm.dbg.value(metadata double %1, metadata !50, metadata !DIExpression()) nounwind, !dbg !38
  tail call void @llvm.dbg.value(metadata i8 %5, metadata !51, metadata !DIExpression()) nounwind, !dbg !38
  %6 = zext i8 %5 to i32, !dbg !39
  %7 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i8* %3, double %1, i32 %6) nounwind, !dbg !39
  %8 = tail call i32 @printer(i8* %3, double %1, i8 zeroext %5) nounwind, !dbg !40
  ret i32 0, !dbg !41
}

declare i32 @puts(i8* nocapture) nounwind

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!48}

!0 = distinct !DISubprogram(name: "printer", linkageName: "printer", line: 12, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 12, file: !46, scope: !1, type: !3, retainedNodes: !43)
!1 = !DIFile(filename: "a.c", directory: "/tmp/")
!2 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "(LLVM build 00)", isOptimized: true, emissionKind: FullDebug, file: !46, enums: !47, retainedTypes: !47, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{!5, !6, !7, !8}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !46, scope: !1, baseType: null)
!7 = !DIBasicType(tag: DW_TAG_base_type, name: "double", size: 64, align: 32, encoding: DW_ATE_float)
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)
!9 = distinct !DISubprogram(name: "inlineprinter", linkageName: "inlineprinter", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 5, file: !46, scope: !1, type: !3, retainedNodes: !44)
!10 = distinct !DISubprogram(name: "main", linkageName: "main", line: 18, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !2, scopeLine: 18, file: !46, scope: !1, type: !11, retainedNodes: !45)
!11 = !DISubroutineType(types: !12)
!12 = !{!5, !5, !13}
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !46, scope: !1, baseType: !14)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, file: !46, scope: !1, baseType: !15)
!15 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!16 = !DILocalVariable(name: "ptr", line: 11, arg: 1, scope: !0, file: !1, type: !6)
!17 = !DILocalVariable(name: "val", line: 11, arg: 2, scope: !0, file: !1, type: !7)
!18 = !DILocalVariable(name: "c", line: 11, arg: 3, scope: !0, file: !1, type: !8)
!19 = !DILocalVariable(name: "ptr", line: 4, arg: 1, scope: !9, file: !1, type: !6)
!20 = !DILocalVariable(name: "val", line: 4, arg: 2, scope: !9, file: !1, type: !7)
!21 = !DILocalVariable(name: "c", line: 4, arg: 3, scope: !9, file: !1, type: !8)

!49 = !DILocalVariable(name: "ptr", line: 4, scope: !10, file: !1, type: !6)
!50 = !DILocalVariable(name: "val", line: 4, scope: !10, file: !1, type: !7)
!51 = !DILocalVariable(name: "c", line: 4, scope: !10, file: !1, type: !8)

!22 = !DILocalVariable(name: "argc", line: 17, arg: 1, scope: !10, file: !1, type: !5)
!23 = !DILocalVariable(name: "argv", line: 17, arg: 2, scope: !10, file: !1, type: !13)
!24 = !DILocalVariable(name: "dval", line: 19, scope: !25, file: !1, type: !7)
!25 = distinct !DILexicalBlock(line: 18, column: 0, file: !46, scope: !10)
!26 = !DILocation(line: 4, scope: !9)
!27 = !DILocation(line: 6, scope: !28)
!28 = distinct !DILexicalBlock(line: 5, column: 0, file: !46, scope: !9)
!29 = !DILocation(line: 7, scope: !28)
!30 = !DILocation(line: 11, scope: !0)
!31 = !DILocation(line: 13, scope: !32)
!32 = distinct !DILexicalBlock(line: 12, column: 0, file: !46, scope: !0)
!33 = !DILocation(line: 14, scope: !32)
!34 = !DILocation(line: 17, scope: !10)
!35 = !DILocation(line: 19, scope: !25)
!36 = !DILocation(line: 20, scope: !25)
!37 = !DILocation(line: 21, scope: !25)
!38 = !DILocation(line: 4, scope: !10, inlinedAt: !37)
!39 = !DILocation(line: 6, scope: !28, inlinedAt: !37)
!40 = !DILocation(line: 22, scope: !25)
!41 = !DILocation(line: 23, scope: !25)
!43 = !{!16, !17, !18}
!44 = !{!19, !20, !21}
!45 = !{!22, !23, !24}
!46 = !DIFile(filename: "a.c", directory: "/tmp/")
!47 = !{}
!48 = !{i32 1, !"Debug Info Version", i32 3}
