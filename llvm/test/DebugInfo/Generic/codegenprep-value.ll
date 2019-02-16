; RUN: opt -codegenprepare -S %s | FileCheck %s
;
; Generated from the following source with:
; clang -O2 -g -S -emit-llvm -mllvm -stop-after=indirectbr-expand test.cpp
;
; extern void use(int);
; extern int foo(long long);
; 
; void test(int p)
; {
;   int i = p + 4;
;   (void)foo(i);  // sign extension of i
;   if (p)
;     return;
;   use(i);        // use of original i
; }
;
; PR39845: Check that CodeGenPrepare does not drop the reference to a local when it is
;          sign-extended and used later.
;
; CHECK: define{{.*}}test
; CHECK: call{{.*}}dbg.value(metadata i32 %p
; CHECK: call{{.*}}dbg.value(metadata i32 %add
;
define dso_local void @_Z4testi(i32 %p) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %p, metadata !12, metadata !DIExpression()), !dbg !14
  %add = add nsw i32 %p, 4, !dbg !15
  call void @llvm.dbg.value(metadata i32 %add, metadata !13, metadata !DIExpression()), !dbg !16
  %conv = sext i32 %add to i64, !dbg !17
  %call = tail call i32 @_Z3foox(i64 %conv), !dbg !18
  %tobool = icmp eq i32 %p, 0, !dbg !19
  br i1 %tobool, label %if.end, label %cleanup, !dbg !21

if.end:
  tail call void @_Z3usei(i32 %add), !dbg !22
  br label %cleanup, !dbg !23

cleanup:
  ret void, !dbg !23
}

declare dso_local i32 @_Z3foox(i64) local_unnamed_addr
declare dso_local void @_Z3usei(i32) local_unnamed_addr

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (trunk 348209)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/home/test/src")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (trunk 348209)"}
!7 = distinct !DISubprogram(name: "test", linkageName: "_Z4testi", scope: !1, file: !1, line: 4, type: !8, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12, !13}
!12 = !DILocalVariable(name: "p", arg: 1, scope: !7, file: !1, line: 4, type: !10)
!13 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 6, type: !10)
!14 = !DILocation(line: 4, column: 15, scope: !7)
!15 = !DILocation(line: 6, column: 13, scope: !7)
!16 = !DILocation(line: 6, column: 7, scope: !7)
!17 = !DILocation(line: 7, column: 13, scope: !7)
!18 = !DILocation(line: 7, column: 9, scope: !7)
!19 = !DILocation(line: 8, column: 7, scope: !20)
!20 = distinct !DILexicalBlock(scope: !7, file: !1, line: 8, column: 7)
!21 = !DILocation(line: 8, column: 7, scope: !7)
!22 = !DILocation(line: 10, column: 3, scope: !7)
!23 = !DILocation(line: 11, column: 1, scope: !7)
