; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=none -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=all -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=none -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=all -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_const_value	(7)
; CHECK-NEXT: DW_AT_name	("i")

; Source to generate the IR below:
; void f1();
; extern bool b;
; extern int x;
; void test() {
;     // i is a const throughout the whole scope and should
;     // use DW_AT_const_value
;     int i = 7;
;     f1();
;     if (b)
;       f1();
; }
;
; $ clang++ -S -emit-llvm -g -O2 simple.cc
;

@b = external dso_local local_unnamed_addr global i8, align 1

define dso_local void @_Z4testv() local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 7, metadata !11, metadata !DIExpression()), !dbg !13
  tail call void @_Z2f1v(), !dbg !14
  %0 = load i8, ptr @b, align 1, !dbg !15, !tbaa !17, !range !21
  %tobool.not = icmp eq i8 %0, 0, !dbg !15
  br i1 %tobool.not, label %if.end, label %if.then, !dbg !22

if.then:                                          ; preds = %entry
  tail call void @_Z2f1v(), !dbg !23
  br label %if.end, !dbg !23

if.end:                                           ; preds = %if.then, %entry
  ret void, !dbg !24
}

declare !dbg !25 dso_local void @_Z2f1v() local_unnamed_addr

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0 (git@github.com:llvm/llvm-project.git 593cb4655097552ac6d81ce18a2851ae0feb8d3c)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "simple.cc", directory: "/code")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 13.0.0 (git@github.com:llvm/llvm-project.git 593cb4655097552ac6d81ce18a2851ae0feb8d3c)"}
!7 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", scope: !1, file: !1, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 14, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 0, scope: !7)
!14 = !DILocation(line: 15, column: 5, scope: !7)
!15 = !DILocation(line: 16, column: 9, scope: !16)
!16 = distinct !DILexicalBlock(scope: !7, file: !1, line: 16, column: 9)
!17 = !{!18, !18, i64 0}
!18 = !{!"bool", !19, i64 0}
!19 = !{!"omnipotent char", !20, i64 0}
!20 = !{!"Simple C++ TBAA"}
!21 = !{i8 0, i8 2}
!22 = !DILocation(line: 16, column: 9, scope: !7)
!23 = !DILocation(line: 17, column: 7, scope: !16)
!24 = !DILocation(line: 18, column: 1, scope: !7)
!25 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
