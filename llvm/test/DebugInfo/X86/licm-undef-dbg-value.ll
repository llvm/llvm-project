; RUN: opt -licm %s -S | FileCheck %s

; CHECK: for.body:
; CHECK-NEXT: llvm.dbg.value(metadata i8 undef

; The load is loop invariant. Check that we leave an undef dbg.value behind
; when licm sinks the instruction.

; clang reduce.cpp -g -O2 -Xclang -disable-llvm-passes -emit-llvm -o reduce.ll
; opt -simplifycfg -sroa -loop-rotate -o - -S reduce.ll
; cat reduce.cpp
; extern char a;
; extern char b;
; void use(char);
; void fun() {
;   char local = 0;
;   for (;b;)
;     local = a;
;   use(local);
; }

@b = external dso_local global i8, align 1
@a = external dso_local global i8, align 1

define dso_local void @_Z3funv() !dbg !12 {
entry:
  call void @llvm.dbg.value(metadata i8 0, metadata !16, metadata !DIExpression()), !dbg !17
  %0 = load i8, i8* @b, align 1, !dbg !18
  %tobool1 = icmp ne i8 %0, 0, !dbg !18
  br i1 %tobool1, label %for.body.lr.ph, label %for.end, !dbg !24

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body, !dbg !24

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %1 = load i8, i8* @a, align 1, !dbg !25
  call void @llvm.dbg.value(metadata i8 %1, metadata !16, metadata !DIExpression()), !dbg !17
  %2 = load i8, i8* @b, align 1, !dbg !18
  %tobool = icmp ne i8 %2, 0, !dbg !18
  br i1 %tobool, label %for.body, label %for.cond.for.end_crit_edge, !dbg !24, !llvm.loop !26

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %split = phi i8 [ %1, %for.body ]
  br label %for.end, !dbg !24

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %local.0.lcssa = phi i8 [ %split, %for.cond.for.end_crit_edge ], [ 0, %entry ], !dbg !17
  call void @llvm.dbg.value(metadata i8 %local.0.lcssa, metadata !16, metadata !DIExpression()), !dbg !17
  call void @_Z3usec(i8 signext %local.0.lcssa), !dbg !28
  ret void, !dbg !29
}

declare !dbg !4 dso_local void @_Z3usec(i8 signext)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "reduce.cpp", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DISubprogram(name: "use", linkageName: "_Z3usec", scope: !1, file: !1, line: 3, type: !5, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7}
!7 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!8 = !{i32 7, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{i32 1, !"wchar_size", i32 4}
!11 = !{!"clang version 11.0.0"}
!12 = distinct !DISubprogram(name: "fun", linkageName: "_Z3funv", scope: !1, file: !1, line: 4, type: !13, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !{!16}
!16 = !DILocalVariable(name: "local", scope: !12, file: !1, line: 5, type: !7)
!17 = !DILocation(line: 0, scope: !12)
!18 = !DILocation(line: 6, column: 9, scope: !19)
!19 = distinct !DILexicalBlock(scope: !20, file: !1, line: 6, column: 3)
!20 = distinct !DILexicalBlock(scope: !12, file: !1, line: 6, column: 3)
!24 = !DILocation(line: 6, column: 3, scope: !20)
!25 = !DILocation(line: 7, column: 13, scope: !19)
!26 = distinct !{!26, !24, !27}
!27 = !DILocation(line: 7, column: 13, scope: !20)
!28 = !DILocation(line: 8, column: 3, scope: !12)
!29 = !DILocation(line: 9, column: 1, scope: !12)
