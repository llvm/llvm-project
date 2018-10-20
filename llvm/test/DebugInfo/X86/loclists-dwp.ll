; RUN: llc -split-dwarf-file=%t1.dwo -filetype=obj -o %t1.o < %s
; RUN: llc -split-dwarf-file=%t2.dwo -filetype=obj -o %t2.o < %p/../Inputs/loclists-dwp-b.ll 
; RUN: llvm-dwp %t1.o %t2.o -o %t.dwp
; RUN: llvm-dwarfdump -v %t.dwp | FileCheck %s

; Make sure that 2 location lists from different units within a dwp file are 
; dumped correctly. The 2 location lists differ in the length of their address
; ranges.
; 
; Generate both .ll files with clang -S -emit-llvm from the following sources:
; a.cpp:
; void y();
; void a(int i) {
;   y();
;   asm("" : : : "rdi");
; }
;
; b.cpp:
; void b(int i) { asm("" : : : "rdi"); }

; CHECK:      DW_AT_location [DW_FORM_sec_offset]   (0x00000000
; CHECK-NEXT: Addr idx 0 (w/ length 6): DW_OP_reg5 RDI)

; CHECK:      DW_AT_location [DW_FORM_sec_offset]   (0x00000000
; CHECK-NEXT: Addr idx 0 (w/ length 0): DW_OP_reg5 RDI)

target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z1ai(i32 %i) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %i, metadata !12, metadata !DIExpression()), !dbg !13
  tail call void @_Z1yv(), !dbg !14
  tail call void asm sideeffect "", "~{rdi},~{dirflag},~{fpsr},~{flags}"(), !dbg !15, !srcloc !16
  ret void, !dbg !17
}

declare dso_local void @_Z1yv() local_unnamed_addr

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0 (https://git.llvm.org/git/clang.git/ 41055c6168135fe539801799e5c5636247cf0302) (https://git.llvm.org/git/llvm.git/ de0558be123ffbb5b5bd692c17dbd57a75fe684f)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "a.cpp", directory: "/home/test/PRs/PR38990")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0 (https://git.llvm.org/git/clang.git/ 41055c6168135fe539801799e5c5636247cf0302) (https://git.llvm.org/git/llvm.git/ de0558be123ffbb5b5bd692c17dbd57a75fe684f)"}
!7 = distinct !DISubprogram(name: "a", linkageName: "_Z1ai", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "i", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !DILocation(line: 2, column: 12, scope: !7)
!14 = !DILocation(line: 3, column: 3, scope: !7)
!15 = !DILocation(line: 4, column: 3, scope: !7)
!16 = !{i32 41}
!17 = !DILocation(line: 5, column: 1, scope: !7)
