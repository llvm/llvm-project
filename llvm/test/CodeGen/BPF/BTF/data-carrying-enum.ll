; RUN: llc -mtriple=bpfel -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
; RUN: llc -mtriple=bpfeb -filetype=obj -o %t1 %s
; RUN: llvm-objcopy --dump-section='.BTF'=%t2 %t1
; RUN: %python %p/print_btf.py %t2 | FileCheck -check-prefixes=CHECK-BTF %s
;
; Source:
;   #![no_std]
;   #![no_main]
;   
;   pub enum DataCarryingEnum {
;       First { a: u32, b: i32 },
;       Second(u32, i32),
;   }
;   
;   #[unsafe(no_mangle)]
;   pub static X: DataCarryingEnum = DataCarryingEnum::First { a: 54, b: -23 };
;   #[unsafe(no_mangle)]
;   pub static Y: DataCarryingEnum = DataCarryingEnum::Second(54, -23);
;   
;   #[cfg(not(test))]
;   #[panic_handler]
;   fn panic(_info: &core::panic::PanicInfo) -> ! {
;       loop {}
;   }
; Compilation flag:
;   cargo +nightly rustc -Zbuild-std=core --target=bpfel-unknown-none -- --emit=llvm-bc
;   llvm-extract --glob=X --glob=Y $(find target/ -name "*.bc" | head -n 1) -o data-carrying-enum.bc
;   llvm-dis data-carrying.bc -o data-carrying-enum.ll

; ModuleID = 'data_carrying.bc'
source_filename = "c0znihgkvro8hs0n88fgrtg6x"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpfel"

@X = constant [12 x i8] c"\00\00\00\006\00\00\00\E9\FF\FF\FF", align 4, !dbg !0
@Y = constant [12 x i8] c"\01\00\00\006\00\00\00\E9\FF\FF\FF", align 4, !dbg !23

!llvm.module.flags = !{!25, !26, !27, !28}
!llvm.ident = !{!29}
!llvm.dbg.cu = !{!30}

; CHECK-BTF:      [1] STRUCT 'DataCarryingEnum' size=12 vlen=1
; CHECK-BTF-NEXT:         '(anon)' type_id=2 bits_offset=0
; CHECK-BTF-NEXT: [2] STRUCT '(anon)' size=12 vlen=2
; CHECK-BTF-NEXT:         'First' type_id=3 bits_offset=0
; CHECK-BTF-NEXT:         'Second' type_id=6 bits_offset=0
; CHECK-BTF-NEXT: [3] STRUCT 'First' size=12 vlen=2
; CHECK-BTF-NEXT:         'a' type_id=4 bits_offset=32
; CHECK-BTF-NEXT:         'b' type_id=5 bits_offset=64
; CHECK-BTF-NEXT: [4] INT 'u32' size=4 bits_offset=0 nr_bits=32 encoding=(none)
; CHECK-BTF-NEXT: [5] INT 'i32' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF-NEXT: [6] STRUCT 'Second' size=12 vlen=2
; CHECK-BTF-NEXT:         '__0' type_id=4 bits_offset=32
; CHECK-BTF-NEXT:         '__1' type_id=5 bits_offset=64
; CHECK-BTF-NEXT: [7] VAR 'X' type_id=2, linkage=global
; CHECK-BTF-NEXT: [8] VAR 'Y' type_id=1, linkage=global
; CHECK-BTF-NEXT: [9] DATASEC '.rodata' size=0 vlen=2
; CHECK-BTF-NEXT:         type_id=7 offset=0 size=12
; CHECK-BTF-NEXT:         type_id=8 offset=0 size=12

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "X", scope: !2, file: !3, line: 10, type: !4, isLocal: false, isDefinition: true, align: 32)
!2 = !DINamespace(name: "data_carrying", scope: null)
!3 = !DIFile(filename: "data-carrying/src/main.rs", directory: "/tmp", checksumkind: CSK_MD5, checksum: "f29b44fc736132241f13c7e516138be1")
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "DataCarryingEnum", scope: !2, file: !5, size: 96, align: 32, flags: DIFlagPublic, elements: !6, templateParams: !16, identifier: "96a1ab4a6d28bee4c2fa742aef7e919c")
!5 = !DIFile(filename: "<unknown>", directory: "")
!6 = !{!7}
!7 = !DICompositeType(tag: DW_TAG_variant_part, scope: !4, file: !5, size: 96, align: 32, elements: !8, templateParams: !16, identifier: "c0e9f845d0e9e3c749154e977aef8d86", discriminator: !22)
!8 = !{!9, !17}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "First", scope: !7, file: !5, baseType: !10, size: 96, align: 32, extraData: i32 0)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "First", scope: !4, file: !5, size: 96, align: 32, flags: DIFlagPublic, elements: !11, templateParams: !16, identifier: "447ce407272d54c94103b4af719086a7")
!11 = !{!12, !14}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !5, baseType: !13, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!13 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !5, baseType: !15, size: 32, align: 32, offset: 64, flags: DIFlagPublic)
!15 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!16 = !{}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "Second", scope: !7, file: !5, baseType: !18, size: 96, align: 32, extraData: i32 1)
!18 = !DICompositeType(tag: DW_TAG_structure_type, name: "Second", scope: !4, file: !5, size: 96, align: 32, flags: DIFlagPublic, elements: !19, templateParams: !16, identifier: "2b27191faeb235d9f91db72ff26e35f7")
!19 = !{!20, !21}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !18, file: !5, baseType: !13, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "__1", scope: !18, file: !5, baseType: !15, size: 32, align: 32, offset: 64, flags: DIFlagPublic)
!22 = !DIDerivedType(tag: DW_TAG_member, scope: !4, file: !5, baseType: !13, size: 32, align: 32, flags: DIFlagArtificial)
!23 = !DIGlobalVariableExpression(var: !24, expr: !DIExpression())
!24 = distinct !DIGlobalVariable(name: "Y", scope: !2, file: !3, line: 12, type: !4, isLocal: false, isDefinition: true, align: 32)
!25 = !{i32 8, !"PIC Level", i32 2}
!26 = !{i32 7, !"PIE Level", i32 2}
!27 = !{i32 7, !"Dwarf Version", i32 4}
!28 = !{i32 2, !"Debug Info Version", i32 3}
!29 = !{!"rustc version 1.91.0-nightly (160e7623e 2025-08-26)"}
!30 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !31, producer: "clang LLVM (rustc version 1.91.0-nightly (160e7623e 2025-08-26))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !32, splitDebugInlining: false, nameTableKind: None)
!31 = !DIFile(filename: "data-carrying-ebpf/src/main.rs/@/c0znihgkvro8hs0n88fgrtg6x", directory: "/tmp")
!32 = !{!0, !23}
