; RUN: llc -dwarf-version=5 -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck %s

; Make sure we don't generate a duplicate DW_AT_rnglists_base attribute in the CU DIE
; when more than one range list is emitted.
; From the source:
;
; void f1();
; void f2() {
;   f1();
;   {
;     bool b;
;     f1();
;     f1();
;   }
; }
; __attribute__((section(".text.foo"))) void f3() { }
 
; Compile with clang -gwarf5 -S -emit-llvm
; and change the resulting IR to move the first call to f1() to between
; the second and the third call. This, along with the dbg.declare instruction
; for the variable b, creates a gap in the code range for the nested lexical
; scope that is b's immediate parent scope. A range list is emitted for that scope.
;
; In addition, the placement of f3() in a different section forces a 
; rangelist to be emitted for the CU instead of simply using low/high pc 
; attributes.

; CHECK:      .debug_info contents:
; CHECK:      DW_TAG_compile_unit
; Make sure we have 2 CU ranges.
; CHECK:      DW_AT_ranges
; CHECK-NEXT: [0x{{[0-9a-f]+, 0x[0-9a-f]+}}) ".text"
; CHECK-NEXT: [0x{{[0-9a-f]+, 0x[0-9a-f]+}}) ".text.foo"
; We should not see any duplicate DW_AT_rnglists_base attributes.
; CHECK:      DW_AT_rnglists_base [DW_FORM_sec_offset]                   (0x0000000c)
; CHECK-NOT:  DW_AT_rnglists_base
;
; Make sure we see 2 ranges in the lexical block DIE.
; CHECK:      DW_TAG_lexical_block
; CHECK-NOT:  DW_TAG
; CHECK:      DW_AT_ranges
; CHECK-NEXT: [0x{{[0-9a-f]+, 0x[0-9a-f]+}}) ".text"
; CHECK-NEXT: [0x{{[0-9a-f]+, 0x[0-9a-f]+}}) ".text"

define dso_local void @_Z2f2v() !dbg !7 {
entry:
  %b = alloca i8, align 1
  call void @llvm.dbg.declare(metadata ptr %b, metadata !11, metadata !DIExpression()), !dbg !14
  call void @_Z2f1v(), !dbg !15
  ; The following call has been moved here from right after the alloca
  call void @_Z2f1v(), !dbg !10  
  call void @_Z2f1v(), !dbg !16
  ret void, !dbg !17
}

declare dso_local void @_Z2f1v()

declare void @llvm.dbg.declare(metadata, metadata, metadata)

define dso_local void @_Z2f3v() section ".text.foo" !dbg !18 {
entry:
  ret void, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 7.0.0 (trunk 337837)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t3.cpp", directory: "/home/test/rangelists", checksumkind: CSK_MD5, checksum: "1ba81b564a832caa8114cd008c199048")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 7.0.0 (trunk 337837)"}
!7 = distinct !DISubprogram(name: "f2", linkageName: "_Z2f2v", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocation(line: 3, column: 3, scope: !7)
!11 = !DILocalVariable(name: "b", scope: !12, file: !1, line: 5, type: !13)
!12 = distinct !DILexicalBlock(scope: !7, file: !1, line: 4, column: 3)
!13 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!14 = !DILocation(line: 5, column: 10, scope: !12)
!15 = !DILocation(line: 6, column: 5, scope: !12)
!16 = !DILocation(line: 7, column: 5, scope: !12)
!17 = !DILocation(line: 9, column: 1, scope: !7)
!18 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !1, file: !1, line: 10, type: !8, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!19 = !DILocation(line: 10, column: 51, scope: !18)
