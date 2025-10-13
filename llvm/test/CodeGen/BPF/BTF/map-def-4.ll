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
;   use core::ptr;
;
;   #[allow(dead_code)]
;   pub struct HashMap<K, V> {
;       key: *const K,
;       value: *const V,
;   }
;
;   impl<K, V> HashMap<K, V> {
;       pub const fn new() -> Self {
;           Self {
;               key: ptr::null(),
;               value: ptr::null(),
;           }
;       }
;   }
;
;   #[unsafe(link_section = ".maps")]
;   #[unsafe(no_mangle)]
;   pub static mut MY_MAP: HashMap<u32, u32> = HashMap::new();
;
;   #[cfg(not(test))]
;   #[panic_handler]
;   fn panic(_info: &core::panic::PanicInfo) -> ! {
;       loop {}
;   }
; Compilation flag:
;   cargo +nightly rustc -Zbuild-std=core --target=bpfel-unknown-none -- --emit=llvm-bc
;   llvm-extract --glob=MY_MAP $(find target/ -name "*.bc" | head -n 1) -o map-def-named-2.bc
;   llvm-dis map-def-named-2.bc -o map-def-named-2.ll

; ModuleID = 'map-def-named-2.bc'
source_filename = "9xlybfhcys3n1sozp3bnamiuu"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpfel"

@MY_MAP = global [16 x i8] zeroinitializer, section ".maps", align 8, !dbg !0

!llvm.module.flags = !{!14, !15, !16, !17}
!llvm.ident = !{!18}
!llvm.dbg.cu = !{!19}

; CHECK-BTF:      [1] PTR '(anon)' type_id=2
; CHECK-BTF-NEXT: [2] INT 'u32' size=4 bits_offset=0 nr_bits=32 encoding=(none)
; CHECK-BTF-NEXT: [3] STRUCT '(anon)' size=16 vlen=2
; CHECK-BTF-NEXT:         'key' type_id=1 bits_offset=0
; CHECK-BTF-NEXT:         'value' type_id=1 bits_offset=64
; CHECK-BTF-NEXT: [4] VAR 'MY_MAP' type_id=3, linkage=global
; CHECK-BTF-NEXT: [5] DATASEC '.maps' size=0 vlen=1
; CHECK-BTF-NEXT:         type_id=4 offset=0 size=16

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "MY_MAP", scope: !2, file: !3, line: 23, type: !4, isLocal: false, isDefinition: true, align: 64)
!2 = !DINamespace(name: "btf_map", scope: null)
!3 = !DIFile(filename: "src/main.rs", directory: "/home/vad/playground/btf-map", checksumkind: CSK_MD5, checksum: "954834fe667cc8cedd8b47ffcd2b489f")
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "HashMap<u32, u32>", scope: !2, file: !5, size: 128, align: 64, flags: DIFlagPublic, elements: !6, templateParams: !11, identifier: "53d93ef1bb5578628ce008544cf10207")
!5 = !DIFile(filename: "<unknown>", directory: "")
!6 = !{!7, !10}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "key", scope: !4, file: !5, baseType: !8, size: 64, align: 64, flags: DIFlagPrivate)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const u32", baseType: !9, size: 64, align: 64, dwarfAddressSpace: 0)
!9 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !4, file: !5, baseType: !8, size: 64, align: 64, offset: 64, flags: DIFlagPrivate)
!11 = !{!12, !13}
!12 = !DITemplateTypeParameter(name: "K", type: !9)
!13 = !DITemplateTypeParameter(name: "V", type: !9)
!14 = !{i32 8, !"PIC Level", i32 2}
!15 = !{i32 7, !"PIE Level", i32 2}
!16 = !{i32 7, !"Dwarf Version", i32 4}
!17 = !{i32 2, !"Debug Info Version", i32 3}
!18 = !{!"rustc version 1.91.0-nightly (160e7623e 2025-08-26)"}
!19 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !20, producer: "clang LLVM (rustc version 1.91.0-nightly (160e7623e 2025-08-26))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !21, splitDebugInlining: false, nameTableKind: None)
!20 = !DIFile(filename: "src/main.rs/@/9xlybfhcys3n1sozp3bnamiuu", directory: "/home/vad/playground/btf-map")
!21 = !{!0}
