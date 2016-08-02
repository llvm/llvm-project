; RUN: llc < %s - | FileCheck %s
; Radar 9309221
; Test dwarf reg no for s16
;CHECK: super-register DW_OP_regx
;CHECK-NEXT: 264
;CHECK-NEXT: DW_OP_piece
;CHECK-NEXT: 4

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

@.str = private unnamed_addr constant [11 x i8] c"%p %lf %c\0A\00"
@.str1 = private unnamed_addr constant [6 x i8] c"point\00"

define i32 @inlineprinter(i8* %ptr, float %val, i8 zeroext %c) nounwind optsize ssp !dbg !0 {
entry:
  tail call void @llvm.dbg.value(metadata i8* %ptr, i64 0, metadata !8, metadata !DIExpression()), !dbg !24
  tail call void @llvm.dbg.value(metadata float %val, i64 0, metadata !10, metadata !DIExpression()), !dbg !25
  tail call void @llvm.dbg.value(metadata i8 %c, i64 0, metadata !12, metadata !DIExpression()), !dbg !26
  %conv = fpext float %val to double, !dbg !27
  %conv3 = zext i8 %c to i32, !dbg !27
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %conv, i32 %conv3) nounwind optsize, !dbg !27
  ret i32 0, !dbg !29
}

declare i32 @printf(i8* nocapture, ...) nounwind optsize

define i32 @printer(i8* %ptr, float %val, i8 zeroext %c) nounwind optsize noinline ssp !dbg !6 {
entry:
  tail call void @llvm.dbg.value(metadata i8* %ptr, i64 0, metadata !14, metadata !DIExpression()), !dbg !30
  tail call void @llvm.dbg.value(metadata float %val, i64 0, metadata !15, metadata !DIExpression()), !dbg !31
  tail call void @llvm.dbg.value(metadata i8 %c, i64 0, metadata !16, metadata !DIExpression()), !dbg !32
  %conv = fpext float %val to double, !dbg !33
  %conv3 = zext i8 %c to i32, !dbg !33
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %conv, i32 %conv3) nounwind optsize, !dbg !33
  ret i32 0, !dbg !35
}

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind optsize ssp !dbg !7 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata !17, metadata !DIExpression()), !dbg !36
  tail call void @llvm.dbg.value(metadata i8** %argv, i64 0, metadata !18, metadata !DIExpression()), !dbg !37
  %conv = sitofp i32 %argc to double, !dbg !38
  %add = fadd double %conv, 5.555552e+05, !dbg !38
  %conv1 = fptrunc double %add to float, !dbg !38
  tail call void @llvm.dbg.value(metadata float %conv1, i64 0, metadata !22, metadata !DIExpression()), !dbg !38
  %call = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str1, i32 0, i32 0)) nounwind optsize, !dbg !39
  %add.ptr = getelementptr i8, i8* bitcast (i32 (i32, i8**)* @main to i8*), i32 %argc, !dbg !40
  %add5 = add nsw i32 %argc, 97, !dbg !40
  %conv6 = trunc i32 %add5 to i8, !dbg !40
  tail call void @llvm.dbg.value(metadata i8* %add.ptr, i64 0, metadata !58, metadata !DIExpression()) nounwind, !dbg !41
  tail call void @llvm.dbg.value(metadata float %conv1, i64 0, metadata !60, metadata !DIExpression()) nounwind, !dbg !42
  tail call void @llvm.dbg.value(metadata i8 %conv6, i64 0, metadata !62, metadata !DIExpression()) nounwind, !dbg !43
  %conv.i = fpext float %conv1 to double, !dbg !44
  %conv3.i = and i32 %add5, 255, !dbg !44
  %call.i = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i8* %add.ptr, double %conv.i, i32 %conv3.i) nounwind optsize, !dbg !44
  %call14 = tail call i32 @printer(i8* %add.ptr, float %conv1, i8 zeroext %conv6) optsize, !dbg !45
  ret i32 0, !dbg !46
}

declare i32 @puts(i8* nocapture) nounwind optsize

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!53}

!0 = distinct !DISubprogram(name: "inlineprinter", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 5, file: !51, scope: !1, type: !3, variables: !48)
!1 = !DIFile(filename: "a.c", directory: "/private/tmp")
!2 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.0 (trunk 129915)", isOptimized: true, emissionKind: 1, file: !51, enums: !52, retainedTypes: !52, subprograms: !47, imports:  null)
!3 = !DISubroutineType(types: !4)
!4 = !{!5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "printer", line: 12, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 12, file: !51, scope: !1, type: !3, variables: !49)
!7 = distinct !DISubprogram(name: "main", line: 18, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, scopeLine: 18, file: !51, scope: !1, type: !3, variables: !50)
!8 = !DILocalVariable(name: "ptr", line: 4, arg: 1, scope: !0, file: !1, type: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: null)
!10 = !DILocalVariable(name: "val", line: 4, arg: 2, scope: !0, file: !1, type: !11)
!11 = !DIBasicType(tag: DW_TAG_base_type, name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!12 = !DILocalVariable(name: "c", line: 4, arg: 3, scope: !0, file: !1, type: !13)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned char", size: 8, align: 8, encoding: DW_ATE_unsigned_char)

!58 = !DILocalVariable(name: "ptr", line: 4, arg: 1, scope: !0, file: !1, type: !9)
!60 = !DILocalVariable(name: "val", line: 4, arg: 2, scope: !0, file: !1, type: !11)
!62 = !DILocalVariable(name: "c", line: 4, arg: 3, scope: !0, file: !1, type: !13)

!14 = !DILocalVariable(name: "ptr", line: 11, arg: 1, scope: !6, file: !1, type: !9)
!15 = !DILocalVariable(name: "val", line: 11, arg: 2, scope: !6, file: !1, type: !11)
!16 = !DILocalVariable(name: "c", line: 11, arg: 3, scope: !6, file: !1, type: !13)
!17 = !DILocalVariable(name: "argc", line: 17, arg: 1, scope: !7, file: !1, type: !5)
!18 = !DILocalVariable(name: "argv", line: 17, arg: 2, scope: !7, file: !1, type: !19)
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: !20)
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 32, align: 32, scope: !2, baseType: !21)
!21 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!22 = !DILocalVariable(name: "dval", line: 19, scope: !23, file: !1, type: !11)
!23 = distinct !DILexicalBlock(line: 18, column: 1, file: !51, scope: !7)
!24 = !DILocation(line: 4, column: 22, scope: !0)
!25 = !DILocation(line: 4, column: 33, scope: !0)
!26 = !DILocation(line: 4, column: 52, scope: !0)
!27 = !DILocation(line: 6, column: 3, scope: !28)
!28 = distinct !DILexicalBlock(line: 5, column: 1, file: !51, scope: !0)
!29 = !DILocation(line: 7, column: 3, scope: !28)
!30 = !DILocation(line: 11, column: 42, scope: !6)
!31 = !DILocation(line: 11, column: 53, scope: !6)
!32 = !DILocation(line: 11, column: 72, scope: !6)
!33 = !DILocation(line: 13, column: 3, scope: !34)
!34 = distinct !DILexicalBlock(line: 12, column: 1, file: !51, scope: !6)
!35 = !DILocation(line: 14, column: 3, scope: !34)
!36 = !DILocation(line: 17, column: 15, scope: !7)
!37 = !DILocation(line: 17, column: 28, scope: !7)
!38 = !DILocation(line: 19, column: 31, scope: !23)
!39 = !DILocation(line: 20, column: 3, scope: !23)
!40 = !DILocation(line: 21, column: 3, scope: !23)
!41 = !DILocation(line: 4, column: 22, scope: !0, inlinedAt: !40)
!42 = !DILocation(line: 4, column: 33, scope: !0, inlinedAt: !40)
!43 = !DILocation(line: 4, column: 52, scope: !0, inlinedAt: !40)
!44 = !DILocation(line: 6, column: 3, scope: !28, inlinedAt: !40)
!45 = !DILocation(line: 22, column: 3, scope: !23)
!46 = !DILocation(line: 23, column: 1, scope: !23)
!47 = !{!0, !6, !7}
!48 = !{!8, !10, !12}
!49 = !{!14, !15, !16}
!50 = !{!17, !18, !22}
!51 = !DIFile(filename: "a.c", directory: "/private/tmp")
!52 = !{}
!53 = !{i32 1, !"Debug Info Version", i32 3}
