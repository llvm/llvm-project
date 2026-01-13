; In the LoongArch architecture, the .text section is subject to
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

; RUN: rm -rf %t && split-file %s %t && cd %t

; RUN: llc -dwarf-version=5 -split-dwarf-file=foo.dwo -O0 -mtriple=loongarch64-unknown-linux-gnu -filetype=obj relax_dwo_ranges.ll -o %t.o
; RUN: llvm-dwarfdump -v %t.o | FileCheck --check-prefix=DWARF5 %s
; RUN: llvm-dwarfdump --debug-info %t.o > /dev/null 2>&1 | count 0
; RUN: llvm-objdump -h %t.o | FileCheck --check-prefix=HDR %s

; RUN: llc -dwarf-version=4 -split-dwarf-file=foo.dwo -O0 -mtriple=loongarch64-unknown-linux-gnu -filetype=obj relax_dwo_ranges.ll -o %t.o
; RUN: llvm-dwarfdump -v %t.o | FileCheck --check-prefix=DWARF4 %s
; RUN: llvm-dwarfdump --debug-info %t.o > /dev/null 2>&1 | count 0
; RUN: llvm-objdump -h %t.o | FileCheck --check-prefix=HDR %s

; Make sure we don't produce any relocations in any .dwo section
; HDR-NOT: .rela.{{.*}}.dwo

; Ensure that 'square()' function uses indexed start and end addresses
; DWARF5: .debug_info.dwo contents:
; DWARF5: DW_TAG_subprogram
; DWARF5-NEXT: DW_AT_low_pc  [DW_FORM_addrx]    (indexed (00000000) address = 0x0000000000000000 ".text")
; DWARF5-NEXT: DW_AT_high_pc [DW_FORM_addrx]    (indexed (00000001) address = 0x0000000000000040 ".text")
; DWARF5: DW_AT_name {{.*}} "square") 
; DWARF5: DW_TAG_formal_parameter

; HDR-NOT: .rela.{{.*}}.dwo

; Ensure there is no unnecessary addresses in .o file
; DWARF5: .debug_addr contents:
; DWARF5: Addrs: [
; DWARF5-NEXT: 0x0000000000000000
; DWARF5-NEXT: 0x0000000000000040
; DWARF5-NEXT: 0x000000000000005c
; DWARF5-NEXT: 0x000000000000009c
; DWARF5-NEXT: 0x00000000000000e0
; DWARF5-NEXT: ]

; HDR-NOT: .rela.{{.*}}.dwo

; Ensure that 'boo()' and 'main()' use DW_RLE_startx_length and DW_RLE_startx_endx
; entries respectively
; DWARF5: .debug_rnglists.dwo contents:
; DWARF5: ranges:
; DWARF5-NEXT: 0x00000014: [DW_RLE_startx_length]:  0x0000000000000002, 0x0000000000000024 => [0x000000000000005c, 0x0000000000000080)
; DWARF5-NEXT: 0x00000017: [DW_RLE_end_of_list  ]
; DWARF5-NEXT: 0x00000018: [DW_RLE_startx_endx  ]:  0x0000000000000003, 0x0000000000000004 => [0x000000000000009c, 0x00000000000000e0)
; DWARF5-NEXT: 0x0000001b: [DW_RLE_end_of_list  ]
; DWARF5-EMPTY:

; HDR-NOT: .rela.{{.*}}.dwo

; DWARF4: .debug_info.dwo contents:
; DWARF4: DW_TAG_subprogram
; DWARF4-NEXT: DW_AT_low_pc  [DW_FORM_GNU_addr_index]	(indexed (00000000) address = 0x0000000000000000 ".text")
; DWARF4-NEXT: DW_AT_high_pc [DW_FORM_GNU_addr_index] (indexed (00000001) address = 0x0000000000000040 ".text")
; DWARF4: DW_AT_name {{.*}} "square") 

; DWARF4: DW_TAG_subprogram
; DWARF4-NEXT: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000002) address = 0x000000000000005c ".text")
; DWARF4-NEXT: DW_AT_high_pc [DW_FORM_data4]	(0x00000024)
; DWARF4: DW_AT_name {{.*}} "boo") 

; DWARF4: DW_TAG_subprogram
; DWARF4-NEXT: DW_AT_low_pc  [DW_FORM_GNU_addr_index] (indexed (00000003) address = 0x000000000000009c ".text")
; DWARF4-NEXT: DW_AT_high_pc [DW_FORM_GNU_addr_index] (indexed (00000004) address = 0x00000000000000e0 ".text")
; DWARF4: DW_AT_name {{.*}} "main") 

; HDR-NOT: .rela.{{.*}}.dwo

; Ensure there is no unnecessary addresses in .o file
; DWARF4: .debug_addr contents:
; DWARF4: Addrs: [
; DWARF4-NEXT: 0x0000000000000000
; DWARF4-NEXT: 0x0000000000000040
; DWARF4-NEXT: 0x000000000000005c
; DWARF4-NEXT: 0x000000000000009c
; DWARF4-NEXT: 0x00000000000000e0
; DWARF4-NEXT: ]

; HDR-NOT: .rela.{{.*}}.dwo

#--- relax_dwo_ranges.cpp
__attribute__((noinline)) int boo();

int square(int num) {
  int num1 = boo();
  return num1 * num;
}

__attribute__((noinline)) int boo() {
  return 8;
}

int main() {
  int a = 10;
  int squared = square(a);
  return squared;
}

#--- gen
clang -g -S -emit-llvm -gsplit-dwarf --target=loongarch64 -march=loongarch64 -O0 relax_dwo_ranges.cpp -o -

#--- relax_dwo_ranges.ll
; ModuleID = 'relax_dwo_ranges.cpp'
source_filename = "relax_dwo_ranges.cpp"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "loongarch64"

; Function Attrs: mustprogress noinline optnone
define dso_local noundef signext i32 @_Z6squarei(i32 noundef signext %num) #0 !dbg !8 {
entry:
  %num.addr = alloca i32, align 4
  %num1 = alloca i32, align 4
  store i32 %num, ptr %num.addr, align 4
    #dbg_declare(ptr %num.addr, !13, !DIExpression(), !14)
    #dbg_declare(ptr %num1, !15, !DIExpression(), !16)
  %call = call noundef signext i32 @_Z3boov(), !dbg !17
  store i32 %call, ptr %num1, align 4, !dbg !16
  %0 = load i32, ptr %num1, align 4, !dbg !18
  %1 = load i32, ptr %num.addr, align 4, !dbg !19
  %mul = mul nsw i32 %0, %1, !dbg !20
  ret i32 %mul, !dbg !21
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local noundef signext i32 @_Z3boov() #1 !dbg !22 {
entry:
  ret i32 8, !dbg !25
}

; Function Attrs: mustprogress noinline norecurse optnone
define dso_local noundef signext i32 @main() #2 !dbg !26 {
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %squared = alloca i32, align 4
  store i32 0, ptr %retval, align 4
    #dbg_declare(ptr %a, !27, !DIExpression(), !28)
  store i32 10, ptr %a, align 4, !dbg !28
    #dbg_declare(ptr %squared, !29, !DIExpression(), !30)
  %0 = load i32, ptr %a, align 4, !dbg !31
  %call = call noundef signext i32 @_Z6squarei(i32 noundef signext %0), !dbg !32
  store i32 %call, ptr %squared, align 4, !dbg !30
  %1 = load i32, ptr %squared, align 4, !dbg !33
  ret i32 %1, !dbg !34
}

attributes #0 = { mustprogress noinline optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="loongarch64" "target-features"="+64bit,+d,+f,+relax,+ual" }
attributes #1 = { mustprogress noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="loongarch64" "target-features"="+64bit,+d,+f,+relax,+ual" }
attributes #2 = { mustprogress noinline norecurse optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="loongarch64" "target-features"="+64bit,+d,+f,+relax,+ual" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "relax_dwo_ranges.dwo", emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "relax_dwo_ranges.cpp", directory: ".", checksumkind: CSK_MD5, checksum: "ecc4b1fa92df66be7da599933e4a21da")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"direct-access-external-data", i32 0}
!6 = !{i32 7, !"frame-pointer", i32 2}
!7 = !{!"clang"}
!8 = distinct !DISubprogram(name: "square", linkageName: "_Z6squarei", scope: !1, file: !1, line: 3, type: !9, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{}
!13 = !DILocalVariable(name: "num", arg: 1, scope: !8, file: !1, line: 3, type: !11)
!14 = !DILocation(line: 3, column: 16, scope: !8)
!15 = !DILocalVariable(name: "num1", scope: !8, file: !1, line: 4, type: !11)
!16 = !DILocation(line: 4, column: 7, scope: !8)
!17 = !DILocation(line: 4, column: 14, scope: !8)
!18 = !DILocation(line: 5, column: 10, scope: !8)
!19 = !DILocation(line: 5, column: 17, scope: !8)
!20 = !DILocation(line: 5, column: 15, scope: !8)
!21 = !DILocation(line: 5, column: 3, scope: !8)
!22 = distinct !DISubprogram(name: "boo", linkageName: "_Z3boov", scope: !1, file: !1, line: 8, type: !23, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!23 = !DISubroutineType(types: !24)
!24 = !{!11}
!25 = !DILocation(line: 9, column: 3, scope: !22)
!26 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 12, type: !23, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!27 = !DILocalVariable(name: "a", scope: !26, file: !1, line: 13, type: !11)
!28 = !DILocation(line: 13, column: 7, scope: !26)
!29 = !DILocalVariable(name: "squared", scope: !26, file: !1, line: 14, type: !11)
!30 = !DILocation(line: 14, column: 7, scope: !26)
!31 = !DILocation(line: 14, column: 24, scope: !26)
!32 = !DILocation(line: 14, column: 17, scope: !26)
!33 = !DILocation(line: 15, column: 10, scope: !26)
!34 = !DILocation(line: 15, column: 3, scope: !26)
