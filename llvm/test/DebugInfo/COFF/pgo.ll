; RUN: llc < %s -filetype=obj | llvm-readobj --codeview - | FileCheck %s

; CHECK: Compile3Sym {
; CHECK:   Flags [ (0x40000)
; CHECK:     PGO (0x40000)

; CHECK: DisplayName: foo
; CHECK: Kind: S_FRAMEPROC (0x1012)
; CHECK:   ProfileGuidedOptimization (0x40000)
; CHECK:   ValidProfileCounts (0x80000)

; CHECK: DisplayName: foo2
; CHECK: Kind: S_FRAMEPROC (0x1012)
; CHECK:   ProfileGuidedOptimization (0x40000)
; CHECK:   ValidProfileCounts (0x80000)

; CHECK: DisplayName: bar
; CHECK: Kind: S_FRAMEPROC (0x1012)
; CHECK:   ProfileGuidedOptimization (0x40000)
; CHECK:   ValidProfileCounts (0x80000)

; CHECK: DisplayName: main
; CHECK: Kind: S_FRAMEPROC (0x1012)
; CHECK-NOT:   ProfileGuidedOptimization (0x40000)
; CHECK-NOT:   ValidProfileCounts (0x80000)

source_filename = "pgo.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.28.29912"

define dso_local i32 @"?foo@@YAHH@Z"(i32 %b) local_unnamed_addr #1 !dbg !43 !prof !49 {
entry:
  call void @llvm.dbg.value(metadata i32 %b, metadata !48, metadata !DIExpression()), !dbg !50
  %mul = mul nsw i32 %b, 10, !dbg !51
  ret i32 %mul, !dbg !51
}

define dso_local i32 @"?foo2@@YAHH@Z"(i32 %a) local_unnamed_addr #1 !dbg !52 !prof !55 {
entry:
  call void @llvm.dbg.value(metadata i32 %a, metadata !54, metadata !DIExpression()), !dbg !56
  %mul = mul nsw i32 %a, 5, !dbg !57
  ret i32 %mul, !dbg !57
}

define dso_local i32 @"?bar@@YAHH@Z"(i32 %num) local_unnamed_addr #1 !dbg !58 !prof !55 {
entry:
  call void @llvm.dbg.value(metadata i32 undef, metadata !60, metadata !DIExpression()), !dbg !61
  %call = tail call i32 @"?foo@@YAHH@Z"(i32 1) #1, !dbg !62
  %call1 = tail call i32 @"?foo2@@YAHH@Z"(i32 2) #1, !dbg !62
  %mul = mul nsw i32 %call1, %call, !dbg !62
  %call2 = tail call i32 @"?foo2@@YAHH@Z"(i32 3) #1, !dbg !62
  %mul3 = mul nsw i32 %mul, %call2, !dbg !62
  ret i32 %mul3, !dbg !62
}

define dso_local i32 @main(i32 %argc, i8** nocapture readnone %argv) local_unnamed_addr #1 !dbg !63 !annotation !72 {
entry:
  call void @llvm.dbg.value(metadata i8** %argv, metadata !70, metadata !DIExpression()), !dbg !73
  call void @llvm.dbg.value(metadata i32 %argc, metadata !71, metadata !DIExpression()), !dbg !73
  %cmp = icmp eq i32 %argc, 2, !dbg !74
  br i1 %cmp, label %return, label %if.end, !dbg !74

if.end:                                           ; preds = %entry
  %cmp1 = icmp slt i32 %argc, 5, !dbg !75
  br i1 %cmp1, label %if.then2, label %if.else, !dbg !75

if.then2:                                         ; preds = %if.end
  %call = tail call i32 @"?bar@@YAHH@Z"(i32 undef) #1, !dbg !76
  br label %return, !dbg !76

if.else:                                          ; preds = %if.end
  %call3 = tail call i32 @"?foo@@YAHH@Z"(i32 %argc) #1, !dbg !79
  br label %return, !dbg !79

return:                                           ; preds = %entry, %if.else, %if.then2
  %retval.0 = phi i32 [ %call, %if.then2 ], [ %call3, %if.else ], [ 0, %entry ], !dbg !73
  ret i32 %retval.0, !dbg !81
}

declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #1 = { optsize }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7, !8, !9, !38}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "pgo.cpp", directory: "")
!2 = !{}
!5 = !{i32 2, !"CodeView", i32 1}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 2}
!8 = !{i32 7, !"PIC Level", i32 2}
!9 = !{i32 1, !"ProfileSummary", !10}
!10 = !{!11, !12, !13, !14, !15, !16, !17, !18, !19, !20}
!11 = !{!"ProfileFormat", !"InstrProf"}
!12 = !{!"TotalCount", i64 2}
!13 = !{!"MaxCount", i64 1}
!14 = !{!"MaxInternalCount", i64 1}
!15 = !{!"MaxFunctionCount", i64 1}
!16 = !{!"NumCounts", i64 5}
!17 = !{!"NumFunctions", i64 4}
!18 = !{!"IsPartialProfile", i64 0}
!19 = !{!"PartialProfileRatio", double 0.000000e+00}
!20 = !{!"DetailedSummary", !21}
!21 = !{!22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37}
!22 = !{i32 10000, i64 0, i32 0}
!23 = !{i32 100000, i64 0, i32 0}
!24 = !{i32 200000, i64 0, i32 0}
!25 = !{i32 300000, i64 0, i32 0}
!26 = !{i32 400000, i64 0, i32 0}
!27 = !{i32 500000, i64 1, i32 2}
!28 = !{i32 600000, i64 1, i32 2}
!29 = !{i32 700000, i64 1, i32 2}
!30 = !{i32 800000, i64 1, i32 2}
!31 = !{i32 900000, i64 1, i32 2}
!32 = !{i32 950000, i64 1, i32 2}
!33 = !{i32 990000, i64 1, i32 2}
!34 = !{i32 999000, i64 1, i32 2}
!35 = !{i32 999900, i64 1, i32 2}
!36 = !{i32 999990, i64 1, i32 2}
!37 = !{i32 999999, i64 1, i32 2}
!38 = !{i32 5, !"CG Profile", !39}
!39 = !{!40, !41}
!40 = !{i32 (i32)* @"?bar@@YAHH@Z", i32 (i32)* @"?foo@@YAHH@Z", i64 0}
!41 = !{i32 (i32)* @"?bar@@YAHH@Z", i32 (i32)* @"?foo2@@YAHH@Z", i64 0}
!43 = distinct !DISubprogram(name: "foo", linkageName: "?foo@@YAHH@Z", scope: !1, file: !1, line: 2, type: !44, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !47)
!44 = !DISubroutineType(types: !45)
!45 = !{!46, !46}
!46 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!47 = !{!48}
!48 = !DILocalVariable(name: "b", arg: 1, scope: !43, file: !1, line: 2, type: !46)
!49 = !{!"function_entry_count", i64 1}
!50 = !DILocation(line: 0, scope: !43)
!51 = !DILocation(line: 3, scope: !43)
!52 = distinct !DISubprogram(name: "foo2", linkageName: "?foo2@@YAHH@Z", scope: !1, file: !1, line: 5, type: !44, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !53)
!53 = !{!54}
!54 = !DILocalVariable(name: "a", arg: 1, scope: !52, file: !1, line: 5, type: !46)
!55 = !{!"function_entry_count", i64 0}
!56 = !DILocation(line: 0, scope: !52)
!57 = !DILocation(line: 6, scope: !52)
!58 = distinct !DISubprogram(name: "bar", linkageName: "?bar@@YAHH@Z", scope: !1, file: !1, line: 8, type: !44, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !59)
!59 = !{!60}
!60 = !DILocalVariable(name: "num", arg: 1, scope: !58, file: !1, line: 8, type: !46)
!61 = !DILocation(line: 0, scope: !58)
!62 = !DILocation(line: 9, scope: !58)
!63 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 11, type: !64, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !69)
!64 = !DISubroutineType(types: !65)
!65 = !{!46, !46, !66}
!66 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !67, size: 64)
!67 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !68, size: 64)
!68 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!69 = !{!70, !71}
!70 = !DILocalVariable(name: "argv", arg: 2, scope: !63, file: !1, line: 11, type: !66)
!71 = !DILocalVariable(name: "argc", arg: 1, scope: !63, file: !1, line: 11, type: !46)
!72 = !{!"instr_prof_hash_mismatch"}
!73 = !DILocation(line: 0, scope: !63)
!74 = !DILocation(line: 13, scope: !63)
!75 = !DILocation(line: 16, scope: !63)
!76 = !DILocation(line: 17, scope: !77)
!77 = distinct !DILexicalBlock(scope: !78, file: !1, line: 16)
!78 = distinct !DILexicalBlock(scope: !63, file: !1, line: 16)
!79 = !DILocation(line: 19, scope: !80)
!80 = distinct !DILexicalBlock(scope: !78, file: !1, line: 18)
!81 = !DILocation(line: 21, scope: !63)
