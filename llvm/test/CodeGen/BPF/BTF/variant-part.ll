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
;       First { a: u32, b: i32, c: u64, d: i64 },
;       Second(u32, i32),
;   }
;
;   #[unsafe(no_mangle)]
;   pub static X: MyEnum = MyEnum::First {
;       a: 54,
;       b: -23,
;       c: 1_324,
;       d: -2_434,
;   };
;   #[unsafe(no_mangle)]
;   pub static Y: MyEnum = MyEnum::Second(54, -23);
;
;   #[cfg(not(test))]
;   #[panic_handler]
;   fn panic(_info: &core::panic::PanicInfo) -> ! {
;       loop {}
;   }
; Compilation flag:
;   cargo +nightly rustc -Zbuild-std=core --target=bpfel-unknown-none -- --emit=llvm-bc
;   llvm-extract --glob=X --glob=Y $(find target/ -name "*.bc" | head -n 1) -o variant-part.bc
;   llvm-dis variant-part.bc -o variant-part.ll

; ModuleID = 'variant-part.bc'
source_filename = "c0znihgkvro8hs0n88fgrtg6x"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpfel"

@X = constant <{ [12 x i8], [4 x i8], [16 x i8] }> <{ [12 x i8] c"\00\00\00\006\00\00\00\E9\FF\FF\FF", [4 x i8] undef, [16 x i8] c",\05\00\00\00\00\00\00~\F6\FF\FF\FF\FF\FF\FF" }>, align 8, !dbg !0
@Y = constant <{ [12 x i8], [20 x i8] }> <{ [12 x i8] c"\01\00\00\006\00\00\00\E9\FF\FF\FF", [20 x i8] undef }>, align 8, !dbg !27

!llvm.module.flags = !{!29, !30, !31, !32}
!llvm.ident = !{!33}
!llvm.dbg.cu = !{!34}

; CHECK-BTF:      [1] STRUCT 'MyEnum' size=32 vlen=1
; CHECK-BTF-NEXT:         '(anon)' type_id=2 bits_offset=0
; CHECK-BTF-NEXT: [2] UNION '(anon)' size=32 vlen=2
; CHECK-BTF-NEXT:         'First' type_id=3 bits_offset=0
; CHECK-BTF-NEXT:         'Second' type_id=8 bits_offset=0
; CHECK-BTF-NEXT: [3] STRUCT 'First' size=32 vlen=4
; CHECK-BTF-NEXT:         'a' type_id=4 bits_offset=32
; CHECK-BTF-NEXT:         'b' type_id=5 bits_offset=64
; CHECK-BTF-NEXT:         'c' type_id=6 bits_offset=128
; CHECK-BTF-NEXT:         'd' type_id=7 bits_offset=192
; CHECK-BTF-NEXT: [4] INT 'u32' size=4 bits_offset=0 nr_bits=32 encoding=(none)
; CHECK-BTF-NEXT: [5] INT 'i32' size=4 bits_offset=0 nr_bits=32 encoding=SIGNED
; CHECK-BTF-NEXT: [6] INT 'u64' size=8 bits_offset=0 nr_bits=64 encoding=(none)
; CHECK-BTF-NEXT: [7] INT 'i64' size=8 bits_offset=0 nr_bits=64 encoding=SIGNED
; CHECK-BTF-NEXT: [8] STRUCT 'Second' size=32 vlen=2
; CHECK-BTF-NEXT:         '__0' type_id=4 bits_offset=32
; CHECK-BTF-NEXT:         '__1' type_id=5 bits_offset=64
; CHECK-BTF-NEXT: [9] VAR 'X' type_id=1, linkage=global
; CHECK-BTF-NEXT: [10] VAR 'Y' type_id=1, linkage=global
; CHECK-BTF-NEXT: [11] DATASEC '.rodata' size=0 vlen=2
; CHECK-BTF-NEXT:         type_id=9 offset=0 size=32
; CHECK-BTF-NEXT:         type_id=10 offset=0 size=32

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "X", scope: !2, file: !3, line: 10, type: !4, isLocal: false, isDefinition: true, align: 64)
!2 = !DINamespace(name: "variant_part", scope: null)
!3 = !DIFile(filename: "variant-part/src/main.rs", directory: "/tmp/variant-part", checksumkind: CSK_MD5, checksum: "10084925e25ca2b70575d3a0510e80a6")
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyEnum", scope: !2, file: !5, size: 256, align: 64, flags: DIFlagPublic, elements: !6, templateParams: !20, identifier: "e720b78cc0f1e09d59c45648bff04f5f")
!5 = !DIFile(filename: "<unknown>", directory: "")
!6 = !{!7}
!7 = !DICompositeType(tag: DW_TAG_variant_part, scope: !4, file: !5, size: 256, align: 64, elements: !8, templateParams: !20, identifier: "410a1af65242eaaa442ea6f6927392f3", discriminator: !26)
!8 = !{!9, !21}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "First", scope: !7, file: !5, baseType: !10, size: 256, align: 64, extraData: i32 0)
!10 = !DICompositeType(tag: DW_TAG_structure_type, name: "First", scope: !4, file: !5, size: 256, align: 64, flags: DIFlagPublic, elements: !11, templateParams: !20, identifier: "6a46b05c482b8e5f117140bdf0fb1397")
!11 = !{!12, !14, !16, !18}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !10, file: !5, baseType: !13, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!13 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!14 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !10, file: !5, baseType: !15, size: 32, align: 32, offset: 64, flags: DIFlagPublic)
!15 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !10, file: !5, baseType: !17, size: 64, align: 64, offset: 128, flags: DIFlagPublic)
!17 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !10, file: !5, baseType: !19, size: 64, align: 64, offset: 192, flags: DIFlagPublic)
!19 = !DIBasicType(name: "i64", size: 64, encoding: DW_ATE_signed)
!20 = !{}
!21 = !DIDerivedType(tag: DW_TAG_member, name: "Second", scope: !7, file: !5, baseType: !22, size: 256, align: 64, extraData: i32 1)
!22 = !DICompositeType(tag: DW_TAG_structure_type, name: "Second", scope: !4, file: !5, size: 256, align: 64, flags: DIFlagPublic, elements: !23, templateParams: !20, identifier: "ecf787a5313c86e9abc87344781957f")
!23 = !{!24, !25}
!24 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !22, file: !5, baseType: !13, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "__1", scope: !22, file: !5, baseType: !15, size: 32, align: 32, offset: 64, flags: DIFlagPublic)
!26 = !DIDerivedType(tag: DW_TAG_member, scope: !4, file: !5, baseType: !13, size: 32, align: 32, flags: DIFlagArtificial)
!27 = !DIGlobalVariableExpression(var: !28, expr: !DIExpression())
!28 = distinct !DIGlobalVariable(name: "Y", scope: !2, file: !3, line: 17, type: !4, isLocal: false, isDefinition: true, align: 64)
!29 = !{i32 8, !"PIC Level", i32 2}
!30 = !{i32 7, !"PIE Level", i32 2}
!31 = !{i32 7, !"Dwarf Version", i32 4}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{!"rustc version 1.91.0-nightly (160e7623e 2025-08-26)"}
!34 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !35, producer: "clang LLVM (rustc version 1.91.0-nightly (160e7623e 2025-08-26))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !36, splitDebugInlining: false, nameTableKind: None)
!35 = !DIFile(filename: "variant-part/src/main.rs/@/c0znihgkvro8hs0n88fgrtg6x", directory: "/tmp/variant-part")
!36 = !{!0, !27}
