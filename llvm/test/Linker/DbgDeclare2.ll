; This file is used by 2011-08-04-DebugLoc.ll, so it doesn't actually do anything itself
;
; RUN: true

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

define void @test(i32 %argc, i8** %argv) uwtable ssp !dbg !5 {
entry:
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %i = alloca i32, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !14, metadata !DIExpression()), !dbg !15
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !16, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.declare(metadata i32* %i, metadata !17, metadata !DIExpression()), !dbg !20
  store i32 0, i32* %i, align 4, !dbg !20
  br label %for.cond, !dbg !20

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4, !dbg !20
  %1 = load i32, i32* %argc.addr, align 4, !dbg !20
  %cmp = icmp slt i32 %0, %1, !dbg !20
  br i1 %cmp, label %for.body, label %for.end, !dbg !20

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4, !dbg !21
  %idxprom = sext i32 %2 to i64, !dbg !21
  %3 = load i8**, i8*** %argv.addr, align 8, !dbg !21
  %arrayidx = getelementptr inbounds i8*, i8** %3, i64 %idxprom, !dbg !21
  %4 = load i8*, i8** %arrayidx, align 8, !dbg !21
  %call = call i32 @puts(i8* %4), !dbg !21
  br label %for.inc, !dbg !23

for.inc:                                          ; preds = %for.body
  %5 = load i32, i32* %i, align 4, !dbg !20
  %inc = add nsw i32 %5, 1, !dbg !20
  store i32 %inc, i32* %i, align 4, !dbg !20
  br label %for.cond, !dbg !20

for.end:                                          ; preds = %for.cond
  ret void, !dbg !24
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i32 @puts(i8*)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!27}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 173515)", isOptimized: true, emissionKind: FullDebug, file: !25, enums: !2, retainedTypes: !2, globals: !2)
!2 = !{}
!5 = distinct !DISubprogram(name: "print_args", linkageName: "test", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 5, file: !26, scope: null, type: !7, retainedNodes: !2)
!6 = !DIFile(filename: "test.cpp", directory: "/private/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{null, !9, !10}
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !12)
!12 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !13)
!13 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!14 = !DILocalVariable(name: "argc", line: 4, arg: 1, scope: !5, file: !6, type: !9)
!15 = !DILocation(line: 4, scope: !5)
!16 = !DILocalVariable(name: "argv", line: 4, arg: 2, scope: !5, file: !6, type: !10)
!17 = !DILocalVariable(name: "i", line: 6, scope: !18, file: !6, type: !9)
!18 = distinct !DILexicalBlock(line: 6, column: 0, file: !26, scope: !19)
!19 = distinct !DILexicalBlock(line: 5, column: 0, file: !26, scope: !5)
!20 = !DILocation(line: 6, scope: !18)
!21 = !DILocation(line: 8, scope: !22)
!22 = distinct !DILexicalBlock(line: 7, column: 0, file: !26, scope: !18)
!23 = !DILocation(line: 9, scope: !22)
!24 = !DILocation(line: 10, scope: !19)
!25 = !DIFile(filename: "main.cpp", directory: "/private/tmp")
!26 = !DIFile(filename: "test.cpp", directory: "/private/tmp")
!27 = !{i32 1, !"Debug Info Version", i32 3}
