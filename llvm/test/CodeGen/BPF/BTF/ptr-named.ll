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
;   pub struct MyType {
;       ptr: *const u32,
;   }
;
;   impl MyType {
;       pub const fn new() -> Self {
;           let ptr = core::ptr::null();
;           Self { ptr }
;       }
;   }
;
;   unsafe impl Sync for MyType {}
;
;   #[unsafe(no_mangle)]
;   pub static X: MyType = MyType::new();
;
;   #[cfg(not(test))]
;   #[panic_handler]
;   fn panic(_info: &core::panic::PanicInfo) -> ! {
;       loop {}
;   }
; Compilation flag:
;   cargo +nightly rustc -Zbuild-std=core --target=bpfel-unknown-none -- --emit=llvm-bc
;   llvm-extract --glob=X $(find target/ -name "*.bc" | head -n 1) -o ptr-named.bc
;   llvm-dis ptr-named.bc -o ptr-named.ll

; ModuleID = 'ptr-named.bc'
source_filename = "1m2uqe50qkwxmo53ydydvou91"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpfel"

@X = constant [8 x i8] zeroinitializer, align 8, !dbg !0

!llvm.module.flags = !{!11, !12, !13, !14}
!llvm.ident = !{!15}
!llvm.dbg.cu = !{!16}

; CHECK-BTF:      [1] STRUCT 'MyType' size=8 vlen=1
; CHECK-BTF-NEXT:         'ptr' type_id=2 bits_offset=0
; CHECK-BTF-NEXT: [2] PTR '(anon)' type_id=3
; CHECK-BTF-NEXT: [3] INT 'u32' size=4 bits_offset=0 nr_bits=32 encoding=(none)
; CHECK-BTF-NEXT: [4] VAR 'X' type_id=1, linkage=global
; CHECK-BTF-NEXT: [5] DATASEC '.rodata' size=0 vlen=1
; CHECK-BTF-NEXT:         type_id=4 offset=0 size=8

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "X", scope: !2, file: !3, line: 19, type: !4, isLocal: false, isDefinition: true, align: 64)
!2 = !DINamespace(name: "ptr_named", scope: null)
!3 = !DIFile(filename: "ptr-named/src/main.rs", directory: "/tmp/ptr-named", checksumkind: CSK_MD5, checksum: "e37168304600b30cbb5ba168f0384932")
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyType", scope: !2, file: !5, size: 64, align: 64, flags: DIFlagPublic, elements: !6, templateParams: !10, identifier: "7609fa40332dd486922f074276a171c3")
!5 = !DIFile(filename: "<unknown>", directory: "")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !4, file: !5, baseType: !8, size: 64, align: 64, flags: DIFlagPrivate)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const u32", baseType: !9, size: 64, align: 64, dwarfAddressSpace: 0)
!9 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!10 = !{}
!11 = !{i32 8, !"PIC Level", i32 2}
!12 = !{i32 7, !"PIE Level", i32 2}
!13 = !{i32 7, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"rustc version 1.92.0-nightly (c8905eaa6 2025-09-28)"}
!16 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !17, producer: "clang LLVM (rustc version 1.92.0-nightly (c8905eaa6 2025-09-28))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !18, splitDebugInlining: false, nameTableKind: None)
!17 = !DIFile(filename: "ptr-named/src/main.rs/@/1m2uqe50qkwxmo53ydydvou91", directory: "/tmp/ptr-named")
!18 = !{!0}
