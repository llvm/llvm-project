; RUN: llc -split-dwarf-file=foo.dwo -O0 %s -function-sections -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-abbrev %t | FileCheck --check-prefix=FUNCTION-SECTIONS %s
; RUN: llvm-readobj --relocations %t | FileCheck --check-prefix=FUNCTION-SECTIONS-RELOCS %s

; RUN: llc -split-dwarf-file=foo.dwo -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-abbrev %t | FileCheck --check-prefix=NO-FUNCTION-SECTIONS %s

; RUN: llc -dwarf-version=5 -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck --check-prefix=DWARF5 %s

; From:
; int foo (int a) {
;   return a+1;
; }
; int bar (int b) {
;   return b+2;
; }

; With function sections enabled make sure that we have a DW_AT_ranges attribute.
; FUNCTION-SECTIONS: DW_AT_ranges

; Check that we have a relocation against the .debug_ranges section.
; FUNCTION-SECTIONS-RELOCS: R_X86_64_32 .debug_ranges 0x0

; Without function sections enabled make sure that we have no DW_AT_ranges attribute.
; NO-FUNCTION-SECTIONS-NOT: DW_AT_ranges
; NO-FUNCTION-SECTIONS: DW_AT_low_pc DW_FORM_addr
; NO-FUNCTION-SECTIONS-NOT: DW_AT_ranges

; For Dwarf 5 check that we neither generate DW_AT_rnglists_base for the CU DIE nor 
; a .debug_rnglists section. There is only 1 CU range with no scope ranges.
;
; DWARF5:     .debug_info contents:
; DWARF5:     DW_TAG_compile_unit
; DWARF5-NOT: DW_AT_rnglists_base [DW_FORM_sec_offset]   
; DWARF5-NOT: .debug_rnglists contents:

; Function Attrs: nounwind uwtable
define i32 @foo(i32 %a) #0 !dbg !4 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %a.addr, metadata !13, metadata !DIExpression()), !dbg !14
  %0 = load i32, ptr %a.addr, align 4, !dbg !14
  %add = add nsw i32 %0, 1, !dbg !14
  ret i32 %add, !dbg !14
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @bar(i32 %b) #0 !dbg !9 {
entry:
  %b.addr = alloca i32, align 4
  store i32 %b, ptr %b.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %b.addr, metadata !15, metadata !DIExpression()), !dbg !16
  %0 = load i32, ptr %b.addr, align 4, !dbg !16
  %add = add nsw i32 %0, 2, !dbg !16
  ret i32 %add, !dbg !16
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11}
!llvm.ident = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)", isOptimized: false, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "z.c", directory: "/usr/local/google/home/echristo")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "z.c", directory: "/usr/local/google/home/echristo")
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "bar", line: 2, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 2, file: !1, scope: !5, type: !6, retainedNodes: !2)
!10 = !{i32 2, !"Dwarf Version", i32 4}
!11 = !{i32 1, !"Debug Info Version", i32 3}
!12 = !{!"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)"}
!13 = !DILocalVariable(name: "a", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!14 = !DILocation(line: 1, scope: !4)
!15 = !DILocalVariable(name: "b", line: 2, arg: 1, scope: !9, file: !5, type: !8)
!16 = !DILocation(line: 2, scope: !9)
