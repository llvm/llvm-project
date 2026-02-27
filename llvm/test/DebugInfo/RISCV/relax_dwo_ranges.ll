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

; RUN: rm -rf %t && split-file %s %t && cd %t

; RUN: llc -dwarf-version=5 -split-dwarf-file=foo.dwo -O0 -mtriple=riscv64-unknown-linux-gnu -filetype=obj relax_dwo_ranges.ll -o %t.o
; RUN: llvm-dwarfdump -v %t.o | FileCheck --check-prefix=DWARF5 %s
; RUN: llvm-dwarfdump --debug-info %t.o > /dev/null 2>&1 | count 0
; RUN: llvm-objdump -h %t.o | FileCheck --check-prefix=HDR %s

; RUN: llc -dwarf-version=4 -split-dwarf-file=foo.dwo -O0 -mtriple=riscv64-unknown-linux-gnu -filetype=obj relax_dwo_ranges.ll -o %t.o
; RUN: llvm-dwarfdump -v %t.o | FileCheck --check-prefix=DWARF4 %s
; RUN: llvm-dwarfdump --debug-info %t.o > /dev/null 2>&1 | count 0
; RUN: llvm-objdump -h %t.o | FileCheck --check-prefix=HDR %s

; Make sure we don't produce any relocations in any .dwo section
; HDR-NOT: .rela.{{.*}}.dwo

; Ensure that 'square()' function uses indexed start and end addresses
; DWARF5: .debug_info.dwo contents:
; DWARF5: DW_TAG_subprogram
; DWARF5-NEXT: DW_AT_low_pc  [DW_FORM_addrx]    (indexed (00000000) address = 0x0000000000000000 ".text")
; DWARF5-NEXT: DW_AT_high_pc [DW_FORM_addrx]    (indexed (00000001) address = 0x000000000000002c ".text")
; DWARF5: DW_AT_name {{.*}} "square") 
; DWARF5: DW_TAG_formal_parameter

; HDR-NOT: .rela.{{.*}}.dwo

; Ensure there is no unnecessary addresses in .o file
; DWARF5: .debug_addr contents:
; DWARF5: Addrs: [
; DWARF5-NEXT: 0x0000000000000000
; DWARF5-NEXT: 0x000000000000002c
; DWARF5-NEXT: 0x000000000000002c
; DWARF5-NEXT: 0x000000000000003e
; DWARF5-NEXT: 0x000000000000006e
; DWARF5-NEXT: ]

; HDR-NOT: .rela.{{.*}}.dwo

; Ensure that 'boo()' and 'main()' use DW_RLE_startx_length and DW_RLE_startx_endx
; entries respectively
; DWARF5: .debug_rnglists.dwo contents:
; DWARF5: ranges:
; DWARF5-NEXT: 0x00000014: [DW_RLE_startx_length]:  0x0000000000000002, 0x0000000000000012 => [0x000000000000002c, 0x000000000000003e)
; DWARF5-NEXT: 0x00000017: [DW_RLE_end_of_list  ]
; DWARF5-NEXT: 0x00000018: [DW_RLE_startx_endx  ]:  0x0000000000000003, 0x0000000000000004 => [0x000000000000003e, 0x000000000000006e)
; DWARF5-NEXT: 0x0000001b: [DW_RLE_end_of_list  ]
; DWARF5-EMPTY:

; HDR-NOT: .rela.{{.*}}.dwo

; DWARF4: .debug_info.dwo contents:
; DWARF4: DW_TAG_subprogram
; DWARF4-NEXT: DW_AT_low_pc  [DW_FORM_GNU_addr_index]	(indexed (00000000) address = 0x0000000000000000 ".text")
; DWARF4-NEXT: DW_AT_high_pc [DW_FORM_GNU_addr_index] (indexed (00000001) address = 0x000000000000002c ".text")
; DWARF4: DW_AT_name {{.*}} "square") 

; DWARF4: DW_TAG_subprogram
; DWARF4-NEXT: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000002) address = 0x000000000000002c ".text")
; DWARF4-NEXT: DW_AT_high_pc [DW_FORM_data4]	(0x00000012)
; DWARF4: DW_AT_name {{.*}} "boo") 

; DWARF4: DW_TAG_subprogram
; DWARF4-NEXT: DW_AT_low_pc  [DW_FORM_GNU_addr_index] (indexed (00000003) address = 0x000000000000003e ".text")
; DWARF4-NEXT: DW_AT_high_pc [DW_FORM_GNU_addr_index] (indexed (00000004) address = 0x000000000000006e ".text")
; DWARF4: DW_AT_name {{.*}} "main") 

; HDR-NOT: .rela.{{.*}}.dwo

; Ensure there is no unnecessary addresses in .o file
; DWARF4: .debug_addr contents:
; DWARF4: Addrs: [
; DWARF4-NEXT: 0x0000000000000000
; DWARF4-NEXT: 0x000000000000002c
; DWARF4-NEXT: 0x000000000000002c
; DWARF4-NEXT: 0x000000000000003e
; DWARF4-NEXT: 0x000000000000006e
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
clang -g -S -emit-llvm -gsplit-dwarf --target=riscv64 -march=rv64gc -O0 relax_dwo_ranges.cpp -o -

#--- relax_dwo_ranges.ll
; ModuleID = 'relax_dwo_ranges.cpp'
source_filename = "relax_dwo_ranges.cpp"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown"

; Function Attrs: mustprogress noinline optnone
define dso_local noundef signext i32 @_Z6squarei(i32 noundef signext %0) #0 !dbg !10 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
    #dbg_declare(ptr %2, !15, !DIExpression(), !16)
    #dbg_declare(ptr %3, !17, !DIExpression(), !18)
  %4 = call noundef signext i32 @_Z3boov(), !dbg !19
  store i32 %4, ptr %3, align 4, !dbg !18
  %5 = load i32, ptr %3, align 4, !dbg !20
  %6 = load i32, ptr %2, align 4, !dbg !21
  %7 = mul nsw i32 %5, %6, !dbg !22
  ret i32 %7, !dbg !23
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local noundef signext i32 @_Z3boov() #1 !dbg !24 {
  ret i32 8, !dbg !27
}

; Function Attrs: mustprogress noinline norecurse optnone
define dso_local noundef signext i32 @main() #2 !dbg !28 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, ptr %1, align 4
    #dbg_declare(ptr %2, !29, !DIExpression(), !30)
  store i32 10, ptr %2, align 4, !dbg !30
    #dbg_declare(ptr %3, !31, !DIExpression(), !32)
  %4 = load i32, ptr %2, align 4, !dbg !33
  %5 = call noundef signext i32 @_Z6squarei(i32 noundef signext %4), !dbg !34
  store i32 %5, ptr %3, align 4, !dbg !32
  %6 = load i32, ptr %3, align 4, !dbg !35
  ret i32 %6, !dbg !36
}

attributes #0 = { mustprogress noinline optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-svukte,-xqccmp,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-experimental-xrivosvisni,-experimental-xrivosvizip,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-zalasr,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvbc32e,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvkgs,-experimental-zvqdotq,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svvptc,-v,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
attributes #1 = { mustprogress noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-svukte,-xqccmp,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-experimental-xrivosvisni,-experimental-xrivosvizip,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-zalasr,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvbc32e,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvkgs,-experimental-zvqdotq,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svvptc,-v,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }
attributes #2 = { mustprogress noinline norecurse optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv64" "target-features"="+64bit,+a,+c,+d,+f,+i,+m,+relax,+zaamo,+zalrsc,+zca,+zcd,+zicsr,+zifencei,+zmmul,-b,-e,-experimental-p,-experimental-svukte,-xqccmp,-xqcia,-xqciac,-xqcibi,-xqcibm,-xqcicli,-xqcicm,-xqcics,-xqcicsr,-xqciint,-xqciio,-xqcilb,-xqcili,-xqcilia,-xqcilo,-xqcilsm,-xqcisim,-xqcisls,-xqcisync,-experimental-xrivosvisni,-experimental-xrivosvizip,-experimental-xsfmclic,-experimental-xsfsclic,-experimental-zalasr,-experimental-zibi,-experimental-zicfilp,-experimental-zicfiss,-experimental-zvbc32e,-experimental-zvfbfa,-experimental-zvfofp8min,-experimental-zvkgs,-experimental-zvqdotq,-h,-q,-sdext,-sdtrig,-sha,-shcounterenw,-shgatpa,-shlcofideleg,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcntrpmf,-smcsrind,-smctr,-smdbltrp,-smepmp,-smmpm,-smnpm,-smrnmi,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssctr,-ssdbltrp,-ssnpm,-sspm,-ssqosid,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-supm,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-svvptc,-v,-xandesbfhcvt,-xandesperf,-xandesvbfhcvt,-xandesvdot,-xandesvpackfph,-xandesvsinth,-xandesvsintload,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xmipscbop,-xmipscmov,-xmipsexectl,-xmipslsp,-xsfcease,-xsfmm128t,-xsfmm16t,-xsfmm32a16f,-xsfmm32a32f,-xsfmm32a8f,-xsfmm32a8i,-xsfmm32t,-xsfmm64a64f,-xsfmm64t,-xsfmmbase,-xsfvcp,-xsfvfbfexp16e,-xsfvfexp16e,-xsfvfexp32e,-xsfvfexpa,-xsfvfexpa64e,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xsmtvdot,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zabha,-zacas,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zcb,-zce,-zcf,-zclsd,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfh,-zfhmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccamoc,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zihintntl,-zihintpause,-zihpm,-zilsd,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: false, runtimeVersion: 0, splitDebugFilename: "relax_dwo_ranges.dwo", emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: GNU)
!1 = !DIFile(filename: "relax_dwo_ranges.cpp", directory: "/proc/self/cwd", checksumkind: CSK_MD5, checksum: "50a257b0f63ed1a964aff88c3623bf0a")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"target-abi", !"lp64d"}
!6 = !{i32 6, !"riscv-isa", !7}
!7 = !{!"rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0"}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{i32 8, !"SmallDataLimit", i32 0}
!10 = distinct !DISubprogram(name: "square", linkageName: "_Z6squarei", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !13}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !{}
!15 = !DILocalVariable(name: "num", arg: 1, scope: !10, file: !1, line: 3, type: !13)
!16 = !DILocation(line: 3, column: 16, scope: !10)
!17 = !DILocalVariable(name: "num1", scope: !10, file: !1, line: 4, type: !13)
!18 = !DILocation(line: 4, column: 7, scope: !10)
!19 = !DILocation(line: 4, column: 14, scope: !10)
!20 = !DILocation(line: 5, column: 10, scope: !10)
!21 = !DILocation(line: 5, column: 17, scope: !10)
!22 = !DILocation(line: 5, column: 15, scope: !10)
!23 = !DILocation(line: 5, column: 3, scope: !10)
!24 = distinct !DISubprogram(name: "boo", linkageName: "_Z3boov", scope: !1, file: !1, line: 8, type: !25, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!25 = !DISubroutineType(types: !26)
!26 = !{!13}
!27 = !DILocation(line: 9, column: 3, scope: !24)
!28 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 12, type: !25, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!29 = !DILocalVariable(name: "a", scope: !28, file: !1, line: 13, type: !13)
!30 = !DILocation(line: 13, column: 7, scope: !28)
!31 = !DILocalVariable(name: "squared", scope: !28, file: !1, line: 14, type: !13)
!32 = !DILocation(line: 14, column: 7, scope: !28)
!33 = !DILocation(line: 14, column: 24, scope: !28)
!34 = !DILocation(line: 14, column: 17, scope: !28)
!35 = !DILocation(line: 15, column: 10, scope: !28)
!36 = !DILocation(line: 15, column: 3, scope: !28)
