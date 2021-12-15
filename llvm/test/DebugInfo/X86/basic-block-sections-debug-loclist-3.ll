; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=none -filetype=obj -o - -experimental-debug-variable-locations=true | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=4 --basic-block-sections=all -filetype=obj -o - -experimental-debug-variable-locations=true | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=none -filetype=obj -o - -experimental-debug-variable-locations=true | llvm-dwarfdump - | FileCheck %s
; RUN: llc %s -mtriple=x86_64-unknown-linux-gnu --dwarf-version=5 --basic-block-sections=all -filetype=obj -o - -experimental-debug-variable-locations=true| llvm-dwarfdump - | FileCheck %s

; CHECK:      DW_TAG_variable
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_name ("tmp")
; CHECK:      DW_TAG_variable
; CHECK-NEXT: DW_AT_location
; CHECK-NEXT: [0x{{[0-9a-f]+}}, 0x{{[0-9a-f]+}}): DW_OP_reg3
; CHECK-NEXT: DW_AT_name	("i")

; Source to generate the IR below:
; void f1();
; int f2();
; extern int x;
; void test() {
;   int tmp = f2();
;   // non-constant value with a single location description
;   // Shouldn't change with BB-sections
;   int i = tmp;
;   f1();
;   x = i;
; }
; $ clang++ -S -emit-llvm -g -O2 loclist_2.cc


@x = external dso_local local_unnamed_addr global i32, align 4

define dso_local void @_Z4testv() local_unnamed_addr !dbg !7 {
entry:
  %call = tail call i32 @_Z2f2v(), !dbg !14
  call void @llvm.dbg.value(metadata i32 %call, metadata !11, metadata !DIExpression()), !dbg !15
  call void @llvm.dbg.value(metadata i32 %call, metadata !13, metadata !DIExpression()), !dbg !15
  tail call void @_Z2f1v(), !dbg !16
  store i32 %call, i32* @x, align 4, !dbg !17, !tbaa !18
  ret void, !dbg !22
}

declare !dbg !23 dso_local i32 @_Z2f2v() local_unnamed_addr

declare !dbg !26 dso_local void @_Z2f1v() local_unnamed_addr

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0 (git@github.com:llvm/llvm-project.git 593cb4655097552ac6d81ce18a2851ae0feb8d3c)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "loclist_3.cc", directory: "/code")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 13.0.0 (git@github.com:llvm/llvm-project.git 593cb4655097552ac6d81ce18a2851ae0feb8d3c)"}
!7 = distinct !DISubprogram(name: "test", linkageName: "_Z4testv", scope: !1, file: !1, line: 4, type: !8, scopeLine: 4, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11, !13}
!11 = !DILocalVariable(name: "tmp", scope: !7, file: !1, line: 5, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 8, type: !12)
!14 = !DILocation(line: 5, column: 13, scope: !7)
!15 = !DILocation(line: 0, scope: !7)
!16 = !DILocation(line: 9, column: 3, scope: !7)
!17 = !DILocation(line: 10, column: 5, scope: !7)
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C++ TBAA"}
!22 = !DILocation(line: 11, column: 1, scope: !7)
!23 = !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !24, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
!24 = !DISubroutineType(types: !25)
!25 = !{!12}
!26 = !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !8, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !2)
