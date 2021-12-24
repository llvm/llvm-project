; REQUIRES: asserts
; RUN: llc < %s -O1 -mtriple=x86_64-unknown-unknown -o /dev/null -debug-only=selectiondag 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

define i32 @main(i32 %argc, i8** %argv) !dbg !8 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !16, metadata !DIExpression()), !dbg !17
  store i8** %argv, i8*** %argv.addr, align 8
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !18, metadata !DIExpression()), !dbg !19
  %0 = load i8**, i8*** %argv.addr, align 8, !dbg !20
  %1 = load i32, i32* %argc.addr, align 4, !dbg !21
  %idxprom = sext i32 %1 to i64, !dbg !20
  %arrayidx = getelementptr inbounds i8*, i8** %0, i64 %idxprom, !dbg !20
  %2 = load i8*, i8** %arrayidx, align 8, !dbg !20
  %arrayidx1 = getelementptr inbounds i8, i8* %2, i64 0, !dbg !20
  %3 = load i8, i8* %arrayidx1, align 1, !dbg !20
  %conv = sext i8 %3 to i32, !dbg !20

  ; CHECK: X86ISD::RET_FLAG {{.*}}, TargetConstant:i32<0>, Register:i32 $eax, {{.*}}, <stdin>:2:3
  ret i32 %conv, !dbg !22
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.0 (trunk 330296) (llvm/trunk 330298)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "-", directory: "/Users/vsk/src/builds/llvm.org-main-RA")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 7.0.0 (trunk 330296) (llvm/trunk 330298)"}
!8 = distinct !DISubprogram(name: "main", scope: !9, file: !9, line: 1, type: !10, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "<stdin>", directory: "/Users/vsk/src/builds/llvm.org-main-RA")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12, !13}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 64)
!15 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!16 = !DILocalVariable(name: "argc", arg: 1, scope: !8, file: !9, line: 1, type: !12)
!17 = !DILocation(line: 1, column: 14, scope: !8)
!18 = !DILocalVariable(name: "argv", arg: 2, scope: !8, file: !9, line: 1, type: !13)
!19 = !DILocation(line: 1, column: 27, scope: !8)
!20 = !DILocation(line: 2, column: 10, scope: !8)
!21 = !DILocation(line: 2, column: 15, scope: !8)
!22 = !DILocation(line: 2, column: 3, scope: !8)
