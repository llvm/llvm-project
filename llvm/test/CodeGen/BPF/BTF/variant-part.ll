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
;   pub enum MyEnum {
;       First { a: u32, b: i32 },
;       Second(u32),
;   }
;
;   #[unsafe(no_mangle)]
;   pub static X: MyEnum = MyEnum::First { a: 54, b: -23 };
;
;   #[cfg(not(test))]
;   #[panic_handler]
;   fn panic(_info: &core::panic::PanicInfo) -> ! {
;       loop {}
;   }
; Compilation flag:
;   cargo +nightly rustc -Zbuild-std=core --target=bpfel-unknown-none -- --emit=llvm-bc
;   llvm-extract --glob=X $(find target/ -name "*.bc" | head -n 1) -o variant-part.bc
;   llvm-dis variant-part.bc -o variant-part.ll

; ModuleID = 'variant-part.bc'
source_filename = "c0znihgkvro8hs0n88fgrtg6x"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpfel"

@X = constant [12 x i8] c"\00\00\00\006\00\00\00\E9\FF\FF\FF", align 4, !dbg !0

!llvm.module.flags = !{!22, !23, !24, !25}
!llvm.ident = !{!26}
!llvm.dbg.cu = !{!27}

; CHECK-BTF:      [1] STRUCT 'MyEnum' size=12 vlen=1
; CHECK-BTF-NEXT:         '(anon)' type_id=3 bits_offset=0
; CHECK-BTF-NEXT: [2] INT 'u32' size=4 bits_offset=0 nr_bits=32 encoding=(none)
; CHECK-BTF-NEXT: [3] UNION '(anon)' size=12 vlen=3
; CHECK-BTF-NEXT:         '(anon)' type_id=2 bits_offset=0
; CHECK-BTF-NEXT:         'First' type_id=4 bits_offset=0
; CHECK-BTF-NEXT:         'Second' type_id=6 bits_offset=0
; CHECK-BTF-NEXT: [4] STRUCT 'First' size=12 vlen=2
; CHECK-BTF-NEXT:         'a' type_id=2 bits_offset=32
; CHECK-BTF-NEXT:         'b' type_id=5 bits_offset=64
; CHECK-BTF-NEXT: [5] INT 'i32' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF-NEXT: [6] STRUCT 'Second' size=12 vlen=1
; CHECK-BTF-NEXT:         '__0' type_id=2 bits_offset=32
; CHECK-BTF-NEXT: [7] VAR 'X' type_id=1, linkage=global
; CHECK-BTF-NEXT: [8] DATASEC '.rodata' size=0 vlen=1
; CHECK-BTF-NEXT:         type_id=7 offset=0 size=12

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "X", scope: !2, file: !3, line: 10, type: !4, isLocal: false, isDefinition: true, align: 32)
!2 = !DINamespace(name: "variant_part", scope: null)
!3 = !DIFile(filename: "variant-part/src/main.rs", directory: "/tmp/variant-part", checksumkind: CSK_MD5, checksum: "b94cd53886ea8f14cbc116b36bc7dd36")
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyEnum", scope: !2, file: !5, size: 96, align: 32, flags: DIFlagPublic, elements: !6, templateParams: !16, identifier: "faba668fd9f71e9b7cf3b9ac5e8b93cb")
!5 = !DIFile(filename: "<unknown>", directory: "")
!6 = !{!7}
!7 = !DICompositeType(tag: DW_TAG_variant_part, scope: !4, file: !5, size: 96, align: 32, elements: !8, templateParams: !16, identifier: "e4aee046fc86d111657622fdcb8c42f7", discriminator: !21)
!8 = !{!9, !17}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "First", scope: !7, file: !5, baseType: !10, size: 96, align: 32, extraData: i32 0)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "First", scope: !4, file: !5, size: 96, align: 32, flags: DIFlagPublic, elements: !11, templateParams: !16, identifier: "cc7748c842e275452db4205b190c8ff7")
!11 = !{!12, !14}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !5, baseType: !13, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!13 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !5, baseType: !15, size: 32, align: 32, offset: 64, flags: DIFlagPublic)
!15 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!16 = !{}
!17 = !DIDerivedType(tag: DW_TAG_member, name: "Second", scope: !7, file: !5, baseType: !18, size: 96, align: 32, extraData: i32 1)
!18 = !DICompositeType(tag: DW_TAG_structure_type, name: "Second", scope: !4, file: !5, size: 96, align: 32, flags: DIFlagPublic, elements: !19, templateParams: !16, identifier: "a2094b1381f3082d504fbd0903aa7c06")
!19 = !{!20}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !18, file: !5, baseType: !13, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!21 = !DIDerivedType(tag: DW_TAG_member, scope: !4, file: !5, baseType: !13, size: 32, align: 32, flags: DIFlagArtificial)
!22 = !{i32 8, !"PIC Level", i32 2}
!23 = !{i32 7, !"PIE Level", i32 2}
!24 = !{i32 7, !"Dwarf Version", i32 4}
!25 = !{i32 2, !"Debug Info Version", i32 3}
!26 = !{!"rustc version 1.91.0-nightly (160e7623e 2025-08-26)"}
!27 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !28, producer: "clang LLVM (rustc version 1.91.0-nightly (160e7623e 2025-08-26))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !29, splitDebugInlining: false, nameTableKind: None)
!28 = !DIFile(filename: "variant-part/src/main.rs/@/c0znihgkvro8hs0n88fgrtg6x", directory: "/tmp/variant-part")
!29 = !{!0}
