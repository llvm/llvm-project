; RUN: llc -dwarf-version=5 -split-dwarf-file=foo.dwo -O0 %s -mtriple=riscv64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck --check-prefix=DWARF5 %s
; RUN: llvm-dwarfdump --debug-info %t 2> %t.txt
; RUN: FileCheck --input-file=%t.txt %s --check-prefix=RELOCS --allow-empty --implicit-check-not=warning:

; RUN: llc -dwarf-version=4 -split-dwarf-file=foo.dwo -O0 %s -mtriple=riscv64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -v %t | FileCheck --check-prefix=DWARF4 %s
; RUN: llvm-dwarfdump --debug-info %t 2> %t.txt
; RUN: FileCheck --input-file=%t.txt %s --check-prefix=RELOCS --allow-empty --implicit-check-not=warning:
; RUN: llvm-objdump -h %t | FileCheck --check-prefix=HDR %s

; In the RISC-V architecture, the .text section is subject to
; relaxation, meaning the start address of each function can change
; during the linking process. Therefore, the .debug_rnglists.dwo
; section must obtain function's start addresses from the .debug_addr
; section.

; Generally, a function's body can be relaxed (for example, the
; square() and main() functions in this test, which contain call
; instructions). For such code ranges, the linker must place the
; start and end addresses into the .debug_addr section and use
; the DW_RLE_startx_endx entry form in the .debug_rnglists.dwo
; section within the .dwo file.

; However, some functions may not contain any relaxable instructions
; (for example, the boo() function in this test). In these cases,
; it is possible to use the more space-efficient DW_RLE_startx_length
; range entry form.

; From the code:

; __attribute__((noinline)) int boo();

; int square(int num) {
;   int num1 = boo();
;   return num1 * num;
; }

; __attribute__((noinline)) int boo() {
;   return 8;
; }

; int main() {
;   int a = 10;
;   int squared = square(a);
;   return squared;
; }

; compiled with

; clang -g -S -gsplit-dwarf --target=riscv64 -march=rv64gc -O0 relax_dwo_ranges.cpp

; RELOCS-NOT: warning: unexpected relocations for dwo section '.debug_info.dwo'

; Make sure we don't produce any relocations in any .dwo section
; HDR-NOT: .rela.{{.*}}.dwo

; Ensure that 'square()' function uses indexed start and end addresses
; DWARF5: .debug_info.dwo contents:
; DWARF5: DW_TAG_subprogram
; DWARF5-NEXT: DW_AT_low_pc  [DW_FORM_addrx]    (indexed (00000000) address = 0x0000000000000000 ".text")
; DWARF5-NEXT: DW_AT_high_pc [DW_FORM_addrx]    (indexed (00000001) address = 0x0000000000000044 ".text")
; DWARF5: DW_AT_name {{.*}} "square") 
; DWARF5: DW_TAG_formal_parameter

; HDR-NOT: .rela.{{.*}}.dwo

; Ensure there is no unnecessary addresses in .o file
; DWARF5: .debug_addr contents:
; DWARF5: Addrs: [
; DWARF5-NEXT: 0x0000000000000000
; DWARF5-NEXT: 0x0000000000000044
; DWARF5-NEXT: 0x0000000000000046
; DWARF5-NEXT: 0x000000000000006c
; DWARF5-NEXT: 0x00000000000000b0
; DWARF5-NEXT: ]

; HDR-NOT: .rela.{{.*}}.dwo

; Ensure that 'boo()' and 'main()' use DW_RLE_startx_length and DW_RLE_startx_endx
; entries respectively
; DWARF5: .debug_rnglists.dwo contents:
; DWARF5: ranges:
; DWARF5-NEXT: 0x00000014: [DW_RLE_startx_length]:  0x0000000000000002, 0x0000000000000024 => [0x0000000000000046, 0x000000000000006a)
; DWARF5-NEXT: 0x00000017: [DW_RLE_end_of_list  ]
; DWARF5-NEXT: 0x00000018: [DW_RLE_startx_endx  ]:  0x0000000000000003, 0x0000000000000004 => [0x000000000000006c, 0x00000000000000b0)
; DWARF5-NEXT: 0x0000001b: [DW_RLE_end_of_list  ]
; DWARF5-EMPTY:

; HDR-NOT: .rela.{{.*}}.dwo

; DWARF4: .debug_info.dwo contents:
; DWARF4: DW_TAG_subprogram
; DWARF4-NEXT: DW_AT_low_pc  [DW_FORM_GNU_addr_index]	(indexed (00000000) address = 0x0000000000000000 ".text")
; DWARF4-NEXT: DW_AT_high_pc [DW_FORM_GNU_addr_index] (indexed (00000001) address = 0x0000000000000044 ".text")
; DWARF4: DW_AT_name {{.*}} "square") 

; DWARF4: DW_TAG_subprogram
; DWARF4-NEXT: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000002) address = 0x0000000000000046 ".text")
; DWARF4-NEXT: DW_AT_high_pc [DW_FORM_data4]	(0x00000024)
; DWARF4: DW_AT_name {{.*}} "boo") 

; DWARF4: DW_TAG_subprogram
; DWARF4-NEXT: DW_AT_low_pc  [DW_FORM_GNU_addr_index] (indexed (00000003) address = 0x000000000000006c ".text")
; DWARF4-NEXT: DW_AT_high_pc [DW_FORM_GNU_addr_index] (indexed (00000004) address = 0x00000000000000b0 ".text")
; DWARF4: DW_AT_name {{.*}} "main") 

; HDR-NOT: .rela.{{.*}}.dwo

; Ensure there is no unnecessary addresses in .o file
; DWARF4: .debug_addr contents:
; DWARF4: Addrs: [
; DWARF4-NEXT: 0x0000000000000000
; DWARF4-NEXT: 0x0000000000000044
; DWARF4-NEXT: 0x0000000000000046
; DWARF4-NEXT: 0x000000000000006c
; DWARF4-NEXT: 0x00000000000000b0
; DWARF4-NEXT: ]

; HDR-NOT: .rela.{{.*}}.dwo

; Function Attrs: mustprogress noinline optnone
define dso_local noundef signext i32 @_Z6squarei(i32 noundef signext %0) #0 !dbg !11 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
    #dbg_declare(ptr %2, !16, !DIExpression(), !17)
    #dbg_declare(ptr %3, !18, !DIExpression(), !19)
  %4 = call noundef signext i32 @_Z3boov(), !dbg !20
  store i32 %4, ptr %3, align 4, !dbg !19
  %5 = load i32, ptr %3, align 4, !dbg !21
  %6 = load i32, ptr %2, align 4, !dbg !22
  %7 = mul nsw i32 %5, %6, !dbg !23
  ret i32 %7, !dbg !24
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local noundef signext i32 @_Z3boov() #1 !dbg !25 {
  ret i32 8, !dbg !28
}

; Function Attrs: mustprogress noinline norecurse optnone
define dso_local noundef signext i32 @main() #2 !dbg !29 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %1, align 4
    #dbg_declare(ptr %2, !30, !DIExpression(), !31)
  store i32 10, ptr %2, align 4, !dbg !31
    #dbg_declare(ptr %3, !32, !DIExpression(), !33)
  %4 = load i32, ptr %2, align 4, !dbg !34
  %5 = call noundef signext i32 @_Z6squarei(i32 noundef signext %4), !dbg !35
  store i32 %5, ptr %3, align 4, !dbg !33
  %6 = load i32, ptr %3, align 4, !dbg !36
  ret i32 %6, !dbg !37
}

attributes #0 = { mustprogress noinline optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+relax,+f,+d" }
attributes #1 = { mustprogress noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+relax,+f,+d" }
attributes #2 = { mustprogress noinline norecurse optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+relax,+f,+d" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 22.0.0git (git@github.com:dlav-sc/llvm-project.git 972928c7a5fecec79f36c6899f1df779d0a17202)", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "riscv_relax_dwo_ranges.dwo", emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "riscv_relax_dwo_ranges.cpp", directory: "/root/test/dwarf/generate", checksumkind: CSK_MD5, checksum: "ea48d4b4acc770ff327714eaf1348b92")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"target-abi", !"lp64d"}
!6 = !{i32 6, !"riscv-isa", !7}
!7 = !{!"rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0"}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{i32 8, !"SmallDataLimit", i32 0}
!10 = !{!"clang version 22.0.0git (git@github.com:dlav-sc/llvm-project.git 972928c7a5fecec79f36c6899f1df779d0a17202)"}
!11 = distinct !DISubprogram(name: "square", linkageName: "_Z6squarei", scope: !1, file: !1, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !{}
!16 = !DILocalVariable(name: "num", arg: 1, scope: !11, file: !1, line: 3, type: !14)
!17 = !DILocation(line: 3, column: 16, scope: !11)
!18 = !DILocalVariable(name: "num1", scope: !11, file: !1, line: 4, type: !14)
!19 = !DILocation(line: 4, column: 7, scope: !11)
!20 = !DILocation(line: 4, column: 14, scope: !11)
!21 = !DILocation(line: 5, column: 10, scope: !11)
!22 = !DILocation(line: 5, column: 17, scope: !11)
!23 = !DILocation(line: 5, column: 15, scope: !11)
!24 = !DILocation(line: 5, column: 3, scope: !11)
!25 = distinct !DISubprogram(name: "boo", linkageName: "_Z3boov", scope: !1, file: !1, line: 8, type: !26, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!26 = !DISubroutineType(types: !27)
!27 = !{!14}
!28 = !DILocation(line: 9, column: 3, scope: !25)
!29 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 12, type: !26, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
!30 = !DILocalVariable(name: "a", scope: !29, file: !1, line: 13, type: !14)
!31 = !DILocation(line: 13, column: 7, scope: !29)
!32 = !DILocalVariable(name: "squared", scope: !29, file: !1, line: 14, type: !14)
!33 = !DILocation(line: 14, column: 7, scope: !29)
!34 = !DILocation(line: 14, column: 24, scope: !29)
!35 = !DILocation(line: 14, column: 17, scope: !29)
!36 = !DILocation(line: 15, column: 10, scope: !29)
!37 = !DILocation(line: 15, column: 3, scope: !29)
