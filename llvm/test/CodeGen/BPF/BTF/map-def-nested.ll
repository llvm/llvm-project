; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK-SHORT %s
; RUN: llc -mtriple=bpfel -filetype=asm -o - %s | FileCheck -check-prefixes=CHECK %s

; Source code:
;   #![no_std]
;   #![no_main]
;   #![allow(dead_code)]
;
;   pub const BPF_MAP_TYPE_HASH: usize = 1;
;
;   // The real map definition.
;   pub struct HashMapDef<K, V, const M: usize, const F: usize> {
;       r#type: *const [i32; BPF_MAP_TYPE_HASH],
;       key: *const K,
;       value: *const V,
;       max_entries: *const [i32; M],
;       map_flags: *const [i32; F],
;   }
;   impl<K, V, const M: usize, const F: usize> HashMapDef<K, V, M, F> {
;       pub const fn new() -> Self {
;           Self {
;               r#type: &[0i32; BPF_MAP_TYPE_HASH],
;               key: ::core::ptr::null(),
;               value: ::core::ptr::null(),
;               max_entries: &[0i32; M],
;               map_flags: &[0i32; F],
;           }
;       }
;   }
;   // Use `UnsafeCell` to allow mutability by multiple threads.
;   pub struct HashMap<K, V, const M: usize, const F: usize = 0>(
;       core::cell::UnsafeCell<HashMapDef<K, V, M, F>>,
;   );
;   impl<K, V, const M: usize, const F: usize> HashMap<K, V, M, F> {
;       pub const fn new() -> Self {
;           Self(core::cell::UnsafeCell::new(HashMapDef::new()))
;       }
;   }
;   // Mark `HashMap` as thread-safe.
;   unsafe impl<K: Sync, V: Sync, const M: usize, const F: usize> Sync for HashMap<K, V, M, F> {}
;
;   // Define custom structs for key and values.
;   pub struct MyKey(u32);
;   pub struct MyValue(u32);
;
;   #[link_section = ".maps"]
;   #[export_name = "HASH_MAP"]
;   pub static HASH_MAP: HashMap<MyKey, MyValue, 10> = HashMap::new();
;
;   #[panic_handler]
;   fn panic(_info: &core::panic::PanicInfo) -> ! {
;       loop {}
;   }
; Compilation flag:
;   cargo +nightly rustc -Zbuild-std=core --target bpfel-unknown-none -- --emit=llvm-ir

; ModuleID = 'map_def.b515d5aaa59f5dac-cgu.0'
source_filename = "map_def.b515d5aaa59f5dac-cgu.0"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpfel"

@alloc_83ea17bf0c4f4a5a5a13d3ae7955acd0 = private unnamed_addr constant [4 x i8] zeroinitializer, align 4
@alloc_e225df0b9c7f20aa5692610f2f2527d2 = private unnamed_addr constant [40 x i8] zeroinitializer, align 4
@HASH_MAP = local_unnamed_addr global <{ ptr, [16 x i8], ptr, ptr }> <{ ptr @alloc_83ea17bf0c4f4a5a5a13d3ae7955acd0, [16 x i8] zeroinitializer, ptr @alloc_e225df0b9c7f20aa5692610f2f2527d2, ptr inttoptr (i64 4 to ptr) }>, section ".maps", align 8, !dbg !0

; __rustc::rust_begin_unwind
; Function Attrs: nofree norecurse noreturn nosync nounwind memory(none)
define hidden void @_RNvCsksSW6QfmhpE_7___rustc17rust_begin_unwind(ptr noalias nocapture noundef readonly align 8 dereferenceable(24) %_info) unnamed_addr #0 !dbg !56 {
start:
    #dbg_value(ptr %_info, !220, !DIExpression(), !221)
  br label %bb1, !dbg !222

bb1:                                              ; preds = %bb1, %start
  br label %bb1, !dbg !222
}

attributes #0 = { nofree norecurse noreturn nosync nounwind memory(none) "target-cpu"="generic" }

; The resulting BTF should look like:
;   #0: <VOID>
;   #1: <PTR> --> [3]
;   #2: <INT> 'i32' bits:32 off:0 enc:signed
;   #3: <ARRAY> n:1 idx-->[4] val-->[2]
;   #4: <INT> '__ARRAY_SIZE_TYPE__' bits:32 off:0
;   #5: <PTR> --> [6]
;   #6: <STRUCT> 'MyKey' sz:4 n:1
;           #00 '__0' off:0 --> [7]
;   #7: <INT> 'u32' bits:32 off:0
;   #8: <PTR> --> [9]
;   #9: <STRUCT> 'MyValue' sz:4 n:1
;           #00 '__0' off:0 --> [7]
;   #10: <PTR> --> [11]
;   #11: <ARRAY> n:10 idx-->[4] val-->[2]
;   #12: <PTR> --> [13]
;   #13: <ARRAY> n:0 idx-->[4] val-->[2]
;   #14: <STRUCT> 'HashMapDef<map_def::MyKey, map_def::MyValue, 10, 0>' sz:40 n:5
;           #00 'type' off:0 --> [1]
;           #01 'key' off:64 --> [5]
;           #02 'value' off:128 --> [8]
;           #03 'max_entries' off:192 --> [10]
;           #04 'map_flags' off:256 --> [12]
;   #15: <STRUCT> 'UnsafeCell<map_def::HashMapDef<map_def::MyKey, map_def::MyValue, 10, 0>>' sz:40 n:1
;           #00 'value' off:0 --> [14]
;   #16: <STRUCT> 'HashMap<map_def::MyKey, map_def::MyValue, 10, 0>' sz:40 n:1
;           #00 '__0' off:0 --> [15]
;   #17: <VAR> 'HASH_MAP' kind:global-alloc --> [16]
;   #18: <PTR> --> [19]
;   #19: <STRUCT> 'PanicInfo' sz:24 n:4
;           #00 'message' off:0 --> [20]
;           #01 'location' off:64 --> [21]
;           #02 'can_unwind' off:128 --> [22]
;           #03 'force_no_backtrace' off:136 --> [22]
;   #20: <PTR> --> [27]
;   #21: <PTR> --> [28]
;   #22: <INT> 'bool' bits:8 off:0 enc:bool
;   #23: <FUNC_PROTO> r-->[0] n:1
;           #00 '_info' --> [18]
;   #24: <FUNC> 'panic' --> static [23]
;   #25: <DATASEC> '.maps' sz:0 n:1
;           #00 off:0 sz:40 --> [17]
;   #26: <DATASEC> '.rodata' sz:0 n:0
;   #27: <FWD> 'Arguments' kind:struct
;   #28: <FWD> 'Location' kind:struct
;
; Before bug https://github.com/llvm/llvm-project/issues/143361 was fixed, the
; BTF kind of MyKey (#6) and MyValue (#9) would be <FWD> instead of <STRUCT>.
; The main goal of this test is making sure that the full <STRUCT> BTF is
; generated for these types.

; We expect exactly 6 structs:
; * MyKey
; * MyValue
; * HashMapDef<map_def::MyKey, map_def::MyValue, 10, 0>
; * UnsafeCell<map_def::HashMapDef<map_def::MyKey, map_def::MyValue, 10, 0>>
; * HashMap<map_def::MyKey, map_def::MyValue, 10, 0>
; * PanicInfo (comes from the Rust core library)
;
; CHECK-SHORT-COUNT-6: BTF_KIND_STRUCT
; CHECK-SHORT-NOT:     BTF_KIND_STRUCT

; We expect exactly 2 forward declarations:
; * Arguments
; * Location
; Both of them come from Rust core library. These types are not used in fields
; of any structs actually used by the program, so skipping them is absolutely
; fine.
;
; CHECK-SHORT-COUNT-2: BTF_KIND_FWD
; CHECK-SHORT-NOT:     BTF_KIND_FWD

; Assert the whole BTF section.
;
;	CHECK:      .section	.BTF,"",@progbits
; CHECK-NEXT: .short	60319                           # 0xeb9f
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .long	24
; CHECK-NEXT: .long	0
; CHECK-NEXT: .long	568
; CHECK-NEXT: .long	568
; CHECK-NEXT: .long	591
; CHECK-NEXT: .long	1                               # BTF_KIND_PTR(id = 1)
; CHECK-NEXT: .long	33554432                        # 0x2000000
; CHECK-NEXT: .long	3
; CHECK-NEXT: .long	17                              # BTF_KIND_INT(id = 2)
; CHECK-NEXT: .long	16777216                        # 0x1000000
; CHECK-NEXT: .long	4
; CHECK-NEXT: .long	16777248                        # 0x1000020
; CHECK-NEXT: .long	0                               # BTF_KIND_ARRAY(id = 3)
; CHECK-NEXT: .long	50331648                        # 0x3000000
; CHECK-NEXT: .long	0
; CHECK-NEXT: .long	2
; CHECK-NEXT: .long	4
; CHECK-NEXT: .long	1
; CHECK-NEXT: .long	21                              # BTF_KIND_INT(id = 4)
; CHECK-NEXT: .long	16777216                        # 0x1000000
; CHECK-NEXT: .long	4
; CHECK-NEXT: .long	32                              # 0x20
; CHECK-NEXT: .long	41                              # BTF_KIND_PTR(id = 5)
; CHECK-NEXT: .long	33554432                        # 0x2000000
; CHECK-NEXT: .long	6
; CHECK-NEXT: .long	63                              # BTF_KIND_STRUCT(id = 6)
; CHECK-NEXT: .long	67108865                        # 0x4000001
; CHECK-NEXT: .long	4
; CHECK-NEXT: .long	69
; CHECK-NEXT: .long	7
; CHECK-NEXT: .long	0                               # 0x0
; CHECK-NEXT: .long	73                              # BTF_KIND_INT(id = 7)
; CHECK-NEXT: .long	16777216                        # 0x1000000
; CHECK-NEXT: .long	4
; CHECK-NEXT: .long	32                              # 0x20
; CHECK-NEXT: .long	77                              # BTF_KIND_PTR(id = 8)
; CHECK-NEXT: .long	33554432                        # 0x2000000
; CHECK-NEXT: .long	9
; CHECK-NEXT: .long	101                             # BTF_KIND_STRUCT(id = 9)
; CHECK-NEXT: .long	67108865                        # 0x4000001
; CHECK-NEXT: .long	4
; CHECK-NEXT: .long	69
; CHECK-NEXT: .long	7
; CHECK-NEXT: .long	0                               # 0x0
; CHECK-NEXT: .long	109                             # BTF_KIND_PTR(id = 10)
; CHECK-NEXT: .long	33554432                        # 0x2000000
; CHECK-NEXT: .long	11
; CHECK-NEXT: .long	0                               # BTF_KIND_ARRAY(id = 11)
; CHECK-NEXT: .long	50331648                        # 0x3000000
; CHECK-NEXT: .long	0
; CHECK-NEXT: .long	2
; CHECK-NEXT: .long	4
; CHECK-NEXT: .long	10
; CHECK-NEXT: .long	126                             # BTF_KIND_PTR(id = 12)
; CHECK-NEXT: .long	33554432                        # 0x2000000
; CHECK-NEXT: .long	13
; CHECK-NEXT: .long	0                               # BTF_KIND_ARRAY(id = 13)
; CHECK-NEXT: .long	50331648                        # 0x3000000
; CHECK-NEXT: .long	0
; CHECK-NEXT: .long	2
; CHECK-NEXT: .long	4
; CHECK-NEXT: .long	0
; CHECK-NEXT: .long	142                             # BTF_KIND_STRUCT(id = 14)
; CHECK-NEXT: .long	67108869                        # 0x4000005
; CHECK-NEXT: .long	40
; CHECK-NEXT: .long	194
; CHECK-NEXT: .long	1
; CHECK-NEXT: .long	0                               # 0x0
; CHECK-NEXT: .long	199
; CHECK-NEXT: .long	5
; CHECK-NEXT: .long	64                              # 0x40
; CHECK-NEXT: .long	203
; CHECK-NEXT: .long	8
; CHECK-NEXT: .long	128                             # 0x80
; CHECK-NEXT: .long	209
; CHECK-NEXT: .long	10
; CHECK-NEXT: .long	192                             # 0xc0
; CHECK-NEXT: .long	221
; CHECK-NEXT: .long	12
; CHECK-NEXT: .long	256                             # 0x100
; CHECK-NEXT: .long	231                             # BTF_KIND_STRUCT(id = 15)
; CHECK-NEXT: .long	67108865                        # 0x4000001
; CHECK-NEXT: .long	40
; CHECK-NEXT: .long	203
; CHECK-NEXT: .long	14
; CHECK-NEXT: .long	0                               # 0x0
; CHECK-NEXT: .long	304                             # BTF_KIND_STRUCT(id = 16)
; CHECK-NEXT: .long	67108865                        # 0x4000001
; CHECK-NEXT: .long	40
; CHECK-NEXT: .long	69
; CHECK-NEXT: .long	15
; CHECK-NEXT: .long	0                               # 0x0
; CHECK-NEXT: .long	353                             # BTF_KIND_VAR(id = 17)
; CHECK-NEXT: .long	234881024                       # 0xe000000
; CHECK-NEXT: .long	16
; CHECK-NEXT: .long	1
; CHECK-NEXT: .long	362                             # BTF_KIND_PTR(id = 18)
; CHECK-NEXT: .long	33554432                        # 0x2000000
; CHECK-NEXT: .long	19
; CHECK-NEXT: .long	398                             # BTF_KIND_STRUCT(id = 19)
; CHECK-NEXT: .long	67108868                        # 0x4000004
; CHECK-NEXT: .long	24
; CHECK-NEXT: .long	408
; CHECK-NEXT: .long	20
; CHECK-NEXT: .long	0                               # 0x0
; CHECK-NEXT: .long	416
; CHECK-NEXT: .long	21
; CHECK-NEXT: .long	64                              # 0x40
; CHECK-NEXT: .long	425
; CHECK-NEXT: .long	22
; CHECK-NEXT: .long	128                             # 0x80
; CHECK-NEXT: .long	436
; CHECK-NEXT: .long	22
; CHECK-NEXT: .long	136                             # 0x88
; CHECK-NEXT: .long	455                             # BTF_KIND_PTR(id = 20)
; CHECK-NEXT: .long	33554432                        # 0x2000000
; CHECK-NEXT: .long	27
; CHECK-NEXT: .long	477                             # BTF_KIND_PTR(id = 21)
; CHECK-NEXT: .long	33554432                        # 0x2000000
; CHECK-NEXT: .long	28
; CHECK-NEXT: .long	510                             # BTF_KIND_INT(id = 22)
; CHECK-NEXT: .long	16777216                        # 0x1000000
; CHECK-NEXT: .long	1
; CHECK-NEXT: .long	67108872                        # 0x4000008
; CHECK-NEXT: .long	0                               # BTF_KIND_FUNC_PROTO(id = 23)
; CHECK-NEXT: .long	218103809                       # 0xd000001
; CHECK-NEXT: .long	0
; CHECK-NEXT: .long	515
; CHECK-NEXT: .long	18
; CHECK-NEXT: .long	521                             # BTF_KIND_FUNC(id = 24)
; CHECK-NEXT: .long	201326592                       # 0xc000000
; CHECK-NEXT: .long	23
; CHECK-NEXT: .long	558                             # BTF_KIND_DATASEC(id = 25)
; CHECK-NEXT: .long	251658241                       # 0xf000001
; CHECK-NEXT: .long	0
; CHECK-NEXT: .long	17
; CHECK-NEXT: .long	HASH_MAP
; CHECK-NEXT: .long	40
; CHECK-NEXT: .long	564                             # BTF_KIND_DATASEC(id = 26)
; CHECK-NEXT: .long	251658240                       # 0xf000000
; CHECK-NEXT: .long	0
; CHECK-NEXT: .long	572                             # BTF_KIND_FWD(id = 27)
; CHECK-NEXT: .long	117440512                       # 0x7000000
; CHECK-NEXT: .long	0
; CHECK-NEXT: .long	582                             # BTF_KIND_FWD(id = 28)
; CHECK-NEXT: .long	117440512                       # 0x7000000
; CHECK-NEXT: .long	0
; CHECK-NEXT: .byte	0                               # string offset=0
; CHECK-NEXT: .ascii	"*const [i32; 1]"               # string offset=1
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"i32"                           # string offset=17
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"__ARRAY_SIZE_TYPE__"           # string offset=21
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"*const map_def::MyKey"         # string offset=41
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"MyKey"                         # string offset=63
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"__0"                           # string offset=69
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"u32"                           # string offset=73
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"*const map_def::MyValue"       # string offset=77
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"MyValue"                       # string offset=101
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"*const [i32; 10]"              # string offset=109
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"*const [i32; 0]"               # string offset=126
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"HashMapDef<map_def::MyKey, map_def::MyValue, 10, 0>" # string offset=142
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"type"                          # string offset=194
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"key"                           # string offset=199
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"value"                         # string offset=203
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"max_entries"                   # string offset=209
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"map_flags"                     # string offset=221
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"UnsafeCell<map_def::HashMapDef<map_def::MyKey, map_def::MyValue, 10, 0>>" # string offset=231
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"HashMap<map_def::MyKey, map_def::MyValue, 10, 0>" # string offset=304
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"HASH_MAP"                      # string offset=353
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"&core::panic::panic_info::PanicInfo" # string offset=362
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"PanicInfo"                     # string offset=398
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"message"                       # string offset=408
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"location"                      # string offset=416
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"can_unwind"                    # string offset=425
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"force_no_backtrace"            # string offset=436
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"&core::fmt::Arguments"         # string offset=455
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"&core::panic::location::Location" # string offset=477
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"bool"                          # string offset=510
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"_info"                         # string offset=515
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"panic"                         # string offset=521
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	".text"                         # string offset=527
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"/tmp/map-def/src/main.rs"      # string offset=533
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	".maps"                         # string offset=558
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	".rodata"                       # string offset=564
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"Arguments"                     # string offset=572
; CHECK-NEXT: .byte	0
; CHECK-NEXT: .ascii	"Location"                      # string offset=582
; CHECK-NEXT: .byte	0

!llvm.module.flags = !{!48, !49, !50, !51}
!llvm.ident = !{!52}
!llvm.dbg.cu = !{!53}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "HASH_MAP", scope: !2, file: !3, line: 44, type: !4, isLocal: false, isDefinition: true, align: 64)
!2 = !DINamespace(name: "map_def", scope: null)
!3 = !DIFile(filename: "map-def/src/main.rs", directory: "/tmp", checksumkind: CSK_MD5, checksum: "35383664e44590293c26c7c1deaae845")
!4 = !DICompositeType(tag: DW_TAG_structure_type, name: "HashMap<map_def::MyKey, map_def::MyValue, 10, 0>", scope: !2, file: !5, size: 320, align: 64, flags: DIFlagPublic, elements: !6, templateParams: !43, identifier: "2820a8a2378f274e261165b6993c728d")
!5 = !DIFile(filename: "<unknown>", directory: "")
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !4, file: !5, baseType: !8, size: 320, align: 64, flags: DIFlagPrivate)
!8 = !DICompositeType(tag: DW_TAG_structure_type, name: "UnsafeCell<map_def::HashMapDef<map_def::MyKey, map_def::MyValue, 10, 0>>", scope: !9, file: !5, size: 320, align: 64, flags: DIFlagPublic, elements: !11, templateParams: !46, identifier: "653cf9ee253b575adf636112b6368fcc")
!9 = !DINamespace(name: "cell", scope: !10)
!10 = !DINamespace(name: "core", scope: null)
!11 = !{!12}
!12 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !8, file: !5, baseType: !13, size: 320, align: 64, flags: DIFlagPrivate)
!13 = !DICompositeType(tag: DW_TAG_structure_type, name: "HashMapDef<map_def::MyKey, map_def::MyValue, 10, 0>", scope: !2, file: !5, size: 320, align: 64, flags: DIFlagPublic, elements: !14, templateParams: !43, identifier: "b4ae0bdae243395185b7e27dd09d6fb6")
!14 = !{!15, !21, !28, !33, !38}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "type", scope: !13, file: !5, baseType: !16, size: 64, align: 64, flags: DIFlagPrivate)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const [i32; 1]", baseType: !17, size: 64, align: 64, dwarfAddressSpace: 0)
!17 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 32, align: 32, elements: !19)
!18 = !DIBasicType(name: "i32", size: 32, encoding: DW_ATE_signed)
!19 = !{!20}
!20 = !DISubrange(count: 1, lowerBound: 0)
!21 = !DIDerivedType(tag: DW_TAG_member, name: "key", scope: !13, file: !5, baseType: !22, size: 64, align: 64, offset: 64, flags: DIFlagPrivate)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const map_def::MyKey", baseType: !23, size: 64, align: 64, dwarfAddressSpace: 0)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyKey", scope: !2, file: !5, size: 32, align: 32, flags: DIFlagPublic, elements: !24, templateParams: !27, identifier: "422e448c565c00452ddf804bbae6e3b0")
!24 = !{!25}
!25 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !23, file: !5, baseType: !26, size: 32, align: 32, flags: DIFlagPrivate)
!26 = !DIBasicType(name: "u32", size: 32, encoding: DW_ATE_unsigned)
!27 = !{}
!28 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !13, file: !5, baseType: !29, size: 64, align: 64, offset: 128, flags: DIFlagPrivate)
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const map_def::MyValue", baseType: !30, size: 64, align: 64, dwarfAddressSpace: 0)
!30 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyValue", scope: !2, file: !5, size: 32, align: 32, flags: DIFlagPublic, elements: !31, templateParams: !27, identifier: "2d0e9f29699904908d5879f2e019e105")
!31 = !{!32}
!32 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !30, file: !5, baseType: !26, size: 32, align: 32, flags: DIFlagPrivate)
!33 = !DIDerivedType(tag: DW_TAG_member, name: "max_entries", scope: !13, file: !5, baseType: !34, size: 64, align: 64, offset: 192, flags: DIFlagPrivate)
!34 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const [i32; 10]", baseType: !35, size: 64, align: 64, dwarfAddressSpace: 0)
!35 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, size: 320, align: 32, elements: !36)
!36 = !{!37}
!37 = !DISubrange(count: 10, lowerBound: 0)
!38 = !DIDerivedType(tag: DW_TAG_member, name: "map_flags", scope: !13, file: !5, baseType: !39, size: 64, align: 64, offset: 256, flags: DIFlagPrivate)
!39 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const [i32; 0]", baseType: !40, size: 64, align: 64, dwarfAddressSpace: 0)
!40 = !DICompositeType(tag: DW_TAG_array_type, baseType: !18, align: 32, elements: !41)
!41 = !{!42}
!42 = !DISubrange(count: 0, lowerBound: 0)
!43 = !{!44, !45}
!44 = !DITemplateTypeParameter(name: "K", type: !23)
!45 = !DITemplateTypeParameter(name: "V", type: !30)
!46 = !{!47}
!47 = !DITemplateTypeParameter(name: "T", type: !13)
!48 = !{i32 8, !"PIC Level", i32 2}
!49 = !{i32 7, !"PIE Level", i32 2}
!50 = !{i32 7, !"Dwarf Version", i32 4}
!51 = !{i32 2, !"Debug Info Version", i32 3}
!52 = !{!"rustc version 1.89.0-nightly (ccf3198de 2025-06-05)"}
!53 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !54, producer: "clang LLVM (rustc version 1.89.0-nightly (ccf3198de 2025-06-05))", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, globals: !55, splitDebugInlining: false, nameTableKind: None)
!54 = !DIFile(filename: "map-def/src/main.rs/@/map_def.b515d5aaa59f5dac-cgu.0", directory: "/tmp")
!55 = !{!0}
!56 = distinct !DISubprogram(name: "panic", linkageName: "_RNvCsksSW6QfmhpE_7___rustc17rust_begin_unwind", scope: !2, file: !3, line: 48, type: !57, scopeLine: 48, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !53, templateParams: !27, retainedNodes: !219)
!57 = !DISubroutineType(types: !58)
!58 = !{null, !59}
!59 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&core::panic::panic_info::PanicInfo", baseType: !60, size: 64, align: 64, dwarfAddressSpace: 0)
!60 = !DICompositeType(tag: DW_TAG_structure_type, name: "PanicInfo", scope: !61, file: !5, size: 192, align: 64, flags: DIFlagPublic, elements: !63, templateParams: !27, identifier: "9712c8cfe8c4fa62648ce00eae7b62e8")
!61 = !DINamespace(name: "panic_info", scope: !62)
!62 = !DINamespace(name: "panic", scope: !10)
!63 = !{!64, !204, !216, !218}
!64 = !DIDerivedType(tag: DW_TAG_member, name: "message", scope: !60, file: !5, baseType: !65, size: 64, align: 64, flags: DIFlagPrivate)
!65 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&core::fmt::Arguments", baseType: !66, size: 64, align: 64, dwarfAddressSpace: 0)
!66 = !DICompositeType(tag: DW_TAG_structure_type, name: "Arguments", scope: !67, file: !5, size: 384, align: 64, flags: DIFlagPublic, elements: !68, templateParams: !27, identifier: "e14e07169a7307941d88f0705f4c531")
!67 = !DINamespace(name: "fmt", scope: !10)
!68 = !{!69, !82, !126}
!69 = !DIDerivedType(tag: DW_TAG_member, name: "pieces", scope: !66, file: !5, baseType: !70, size: 128, align: 64, flags: DIFlagPrivate)
!70 = !DICompositeType(tag: DW_TAG_structure_type, name: "&[&str]", file: !5, size: 128, align: 64, elements: !71, templateParams: !27, identifier: "4e66b00a376d6af5b8765440fb2839f")
!71 = !{!72, !81}
!72 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !70, file: !5, baseType: !73, size: 64, align: 64)
!73 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !74, size: 64, align: 64, dwarfAddressSpace: 0)
!74 = !DICompositeType(tag: DW_TAG_structure_type, name: "&str", file: !5, size: 128, align: 64, elements: !75, templateParams: !27, identifier: "9277eecd40495f85161460476aacc992")
!75 = !{!76, !79}
!76 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !74, file: !5, baseType: !77, size: 64, align: 64)
!77 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !78, size: 64, align: 64, dwarfAddressSpace: 0)
!78 = !DIBasicType(name: "u8", size: 8, encoding: DW_ATE_unsigned)
!79 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !74, file: !5, baseType: !80, size: 64, align: 64, offset: 64)
!80 = !DIBasicType(name: "usize", size: 64, encoding: DW_ATE_unsigned)
!81 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !70, file: !5, baseType: !80, size: 64, align: 64, offset: 64)
!82 = !DIDerivedType(tag: DW_TAG_member, name: "fmt", scope: !66, file: !5, baseType: !83, size: 128, align: 64, offset: 256, flags: DIFlagPrivate)
!83 = !DICompositeType(tag: DW_TAG_structure_type, name: "Option<&[core::fmt::rt::Placeholder]>", scope: !84, file: !5, size: 128, align: 64, flags: DIFlagPublic, elements: !85, templateParams: !27, identifier: "6e0973f30da8f3e9b2717ca992f0b714")
!84 = !DINamespace(name: "option", scope: !10)
!85 = !{!86}
!86 = !DICompositeType(tag: DW_TAG_variant_part, scope: !83, file: !5, size: 128, align: 64, elements: !87, templateParams: !27, identifier: "35c8646d1aaa5695eb2c3e3264c03de", discriminator: !124)
!87 = !{!88, !120}
!88 = !DIDerivedType(tag: DW_TAG_member, name: "None", scope: !86, file: !5, baseType: !89, size: 128, align: 64, extraData: i64 0)
!89 = !DICompositeType(tag: DW_TAG_structure_type, name: "None", scope: !83, file: !5, size: 128, align: 64, flags: DIFlagPublic, elements: !27, templateParams: !90, identifier: "93e3b0ec85807ee77ffba5ea5f3b28c1")
!90 = !{!91}
!91 = !DITemplateTypeParameter(name: "T", type: !92)
!92 = !DICompositeType(tag: DW_TAG_structure_type, name: "&[core::fmt::rt::Placeholder]", file: !5, size: 128, align: 64, elements: !93, templateParams: !27, identifier: "10c41c2df1315d111a0393731f66c09")
!93 = !{!94, !119}
!94 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !92, file: !5, baseType: !95, size: 64, align: 64)
!95 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !96, size: 64, align: 64, dwarfAddressSpace: 0)
!96 = !DICompositeType(tag: DW_TAG_structure_type, name: "Placeholder", scope: !97, file: !5, size: 384, align: 64, flags: DIFlagPublic, elements: !98, templateParams: !27, identifier: "cad9cf74409b285e595ee58b1e8af246")
!97 = !DINamespace(name: "rt", scope: !67)
!98 = !{!99, !100, !101, !118}
!99 = !DIDerivedType(tag: DW_TAG_member, name: "position", scope: !96, file: !5, baseType: !80, size: 64, align: 64, offset: 256, flags: DIFlagPublic)
!100 = !DIDerivedType(tag: DW_TAG_member, name: "flags", scope: !96, file: !5, baseType: !26, size: 32, align: 32, offset: 320, flags: DIFlagPublic)
!101 = !DIDerivedType(tag: DW_TAG_member, name: "precision", scope: !96, file: !5, baseType: !102, size: 128, align: 64, flags: DIFlagPublic)
!102 = !DICompositeType(tag: DW_TAG_structure_type, name: "Count", scope: !97, file: !5, size: 128, align: 64, flags: DIFlagPublic, elements: !103, templateParams: !27, identifier: "47a3e082ebd2197cdee3557b78db0895")
!103 = !{!104}
!104 = !DICompositeType(tag: DW_TAG_variant_part, scope: !102, file: !5, size: 128, align: 64, elements: !105, templateParams: !27, identifier: "fbfebe02a88a23e6b3fee451fbc7a22a", discriminator: !117)
!105 = !{!106, !111, !115}
!106 = !DIDerivedType(tag: DW_TAG_member, name: "Is", scope: !104, file: !5, baseType: !107, size: 128, align: 64, extraData: i16 0)
!107 = !DICompositeType(tag: DW_TAG_structure_type, name: "Is", scope: !102, file: !5, size: 128, align: 64, flags: DIFlagPublic, elements: !108, templateParams: !27, identifier: "35ea4a205f9e303fadb87098789a413")
!108 = !{!109}
!109 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !107, file: !5, baseType: !110, size: 16, align: 16, offset: 16, flags: DIFlagPublic)
!110 = !DIBasicType(name: "u16", size: 16, encoding: DW_ATE_unsigned)
!111 = !DIDerivedType(tag: DW_TAG_member, name: "Param", scope: !104, file: !5, baseType: !112, size: 128, align: 64, extraData: i16 1)
!112 = !DICompositeType(tag: DW_TAG_structure_type, name: "Param", scope: !102, file: !5, size: 128, align: 64, flags: DIFlagPublic, elements: !113, templateParams: !27, identifier: "128e9b9903f5bfc4171032ebc60831ae")
!113 = !{!114}
!114 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !112, file: !5, baseType: !80, size: 64, align: 64, offset: 64, flags: DIFlagPublic)
!115 = !DIDerivedType(tag: DW_TAG_member, name: "Implied", scope: !104, file: !5, baseType: !116, size: 128, align: 64, extraData: i16 2)
!116 = !DICompositeType(tag: DW_TAG_structure_type, name: "Implied", scope: !102, file: !5, size: 128, align: 64, flags: DIFlagPublic, elements: !27, identifier: "2b5e194069abb0e2cca16baf7c4de2e5")
!117 = !DIDerivedType(tag: DW_TAG_member, scope: !102, file: !5, baseType: !110, size: 16, align: 16, flags: DIFlagArtificial)
!118 = !DIDerivedType(tag: DW_TAG_member, name: "width", scope: !96, file: !5, baseType: !102, size: 128, align: 64, offset: 128, flags: DIFlagPublic)
!119 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !92, file: !5, baseType: !80, size: 64, align: 64, offset: 64)
!120 = !DIDerivedType(tag: DW_TAG_member, name: "Some", scope: !86, file: !5, baseType: !121, size: 128, align: 64)
!121 = !DICompositeType(tag: DW_TAG_structure_type, name: "Some", scope: !83, file: !5, size: 128, align: 64, flags: DIFlagPublic, elements: !122, templateParams: !90, identifier: "969e366aa9d3e1d1fed24b2e095f5e8c")
!122 = !{!123}
!123 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !121, file: !5, baseType: !92, size: 128, align: 64, flags: DIFlagPublic)
!124 = !DIDerivedType(tag: DW_TAG_member, scope: !83, file: !5, baseType: !125, size: 64, align: 64, flags: DIFlagArtificial)
!125 = !DIBasicType(name: "u64", size: 64, encoding: DW_ATE_unsigned)
!126 = !DIDerivedType(tag: DW_TAG_member, name: "args", scope: !66, file: !5, baseType: !127, size: 128, align: 64, offset: 128, flags: DIFlagPrivate)
!127 = !DICompositeType(tag: DW_TAG_structure_type, name: "&[core::fmt::rt::Argument]", file: !5, size: 128, align: 64, elements: !128, templateParams: !27, identifier: "71de5e40af8196ca3c26c6bddd95e9f2")
!128 = !{!129, !203}
!129 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !127, file: !5, baseType: !130, size: 64, align: 64)
!130 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !131, size: 64, align: 64, dwarfAddressSpace: 0)
!131 = !DICompositeType(tag: DW_TAG_structure_type, name: "Argument", scope: !97, file: !5, size: 128, align: 64, flags: DIFlagPublic, elements: !132, templateParams: !27, identifier: "dbc70fce95dc8ac1ce4fe4858fdabe80")
!132 = !{!133}
!133 = !DIDerivedType(tag: DW_TAG_member, name: "ty", scope: !131, file: !5, baseType: !134, size: 128, align: 64, flags: DIFlagPrivate)
!134 = !DICompositeType(tag: DW_TAG_structure_type, name: "ArgumentType", scope: !97, file: !5, size: 128, align: 64, flags: DIFlagPrivate, elements: !135, templateParams: !27, identifier: "7d01910dc70f1a78a8a27405eab8a43a")
!135 = !{!136}
!136 = !DICompositeType(tag: DW_TAG_variant_part, scope: !134, file: !5, size: 128, align: 64, elements: !137, templateParams: !27, identifier: "129fc9475b57da8370ede7dd68862c56", discriminator: !202)
!137 = !{!138, !198}
!138 = !DIDerivedType(tag: DW_TAG_member, name: "Placeholder", scope: !136, file: !5, baseType: !139, size: 128, align: 64)
!139 = !DICompositeType(tag: DW_TAG_structure_type, name: "Placeholder", scope: !134, file: !5, size: 128, align: 64, flags: DIFlagPrivate, elements: !140, templateParams: !27, identifier: "336cbabc928f7615d3930a04f9cd1e7b")
!140 = !{!141, !151, !192}
!141 = !DIDerivedType(tag: DW_TAG_member, name: "value", scope: !139, file: !5, baseType: !142, size: 64, align: 64, flags: DIFlagPrivate)
!142 = !DICompositeType(tag: DW_TAG_structure_type, name: "NonNull<()>", scope: !143, file: !5, size: 64, align: 64, flags: DIFlagPublic, elements: !145, templateParams: !149, identifier: "d53c6809de9a41beeb232183f0e84949")
!143 = !DINamespace(name: "non_null", scope: !144)
!144 = !DINamespace(name: "ptr", scope: !10)
!145 = !{!146}
!146 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !142, file: !5, baseType: !147, size: 64, align: 64, flags: DIFlagPrivate)
!147 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "*const ()", baseType: !148, size: 64, align: 64, dwarfAddressSpace: 0)
!148 = !DIBasicType(name: "()", encoding: DW_ATE_unsigned)
!149 = !{!150}
!150 = !DITemplateTypeParameter(name: "T", type: !148)
!151 = !DIDerivedType(tag: DW_TAG_member, name: "formatter", scope: !139, file: !5, baseType: !152, size: 64, align: 64, offset: 64, flags: DIFlagPrivate)
!152 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "unsafe fn(core::ptr::non_null::NonNull<()>, &mut core::fmt::Formatter) -> core::result::Result<(), core::fmt::Error>", baseType: !153, size: 64, align: 64, dwarfAddressSpace: 0)
!153 = !DISubroutineType(types: !154)
!154 = !{!155, !142, !172}
!155 = !DICompositeType(tag: DW_TAG_structure_type, name: "Result<(), core::fmt::Error>", scope: !156, file: !5, size: 8, align: 8, flags: DIFlagPublic, elements: !157, templateParams: !27, identifier: "99973da8cf54c7f5317c926d59615582")
!156 = !DINamespace(name: "result", scope: !10)
!157 = !{!158}
!158 = !DICompositeType(tag: DW_TAG_variant_part, scope: !155, file: !5, size: 8, align: 8, elements: !159, templateParams: !27, identifier: "53ed6832cf19f5ad2f71c3acd544f2e7", discriminator: !171)
!159 = !{!160, !167}
!160 = !DIDerivedType(tag: DW_TAG_member, name: "Ok", scope: !158, file: !5, baseType: !161, size: 8, align: 8, extraData: i8 0)
!161 = !DICompositeType(tag: DW_TAG_structure_type, name: "Ok", scope: !155, file: !5, size: 8, align: 8, flags: DIFlagPublic, elements: !162, templateParams: !164, identifier: "235c45874b85385ca484be5f3c28dd22")
!162 = !{!163}
!163 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !161, file: !5, baseType: !148, align: 8, offset: 8, flags: DIFlagPublic)
!164 = !{!150, !165}
!165 = !DITemplateTypeParameter(name: "E", type: !166)
!166 = !DICompositeType(tag: DW_TAG_structure_type, name: "Error", scope: !67, file: !5, align: 8, flags: DIFlagPublic, elements: !27, identifier: "2dfe614afd248327fa32734957f7830b")
!167 = !DIDerivedType(tag: DW_TAG_member, name: "Err", scope: !158, file: !5, baseType: !168, size: 8, align: 8, extraData: i8 1)
!168 = !DICompositeType(tag: DW_TAG_structure_type, name: "Err", scope: !155, file: !5, size: 8, align: 8, flags: DIFlagPublic, elements: !169, templateParams: !164, identifier: "f206cecb112e889bb393141c32ac7d8")
!169 = !{!170}
!170 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !168, file: !5, baseType: !166, align: 8, offset: 8, flags: DIFlagPublic)
!171 = !DIDerivedType(tag: DW_TAG_member, scope: !155, file: !5, baseType: !78, size: 8, align: 8, flags: DIFlagArtificial)
!172 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&mut core::fmt::Formatter", baseType: !173, size: 64, align: 64, dwarfAddressSpace: 0)
!173 = !DICompositeType(tag: DW_TAG_structure_type, name: "Formatter", scope: !67, file: !5, size: 192, align: 64, flags: DIFlagPublic, elements: !174, templateParams: !27, identifier: "2c1103f1655ea830c1cfb9fb313231b2")
!174 = !{!175, !181}
!175 = !DIDerivedType(tag: DW_TAG_member, name: "options", scope: !173, file: !5, baseType: !176, size: 64, align: 32, offset: 128, flags: DIFlagPrivate)
!176 = !DICompositeType(tag: DW_TAG_structure_type, name: "FormattingOptions", scope: !67, file: !5, size: 64, align: 32, flags: DIFlagPublic, elements: !177, templateParams: !27, identifier: "d60d0acde5a657976caaaf83c6e67d5a")
!177 = !{!178, !179, !180}
!178 = !DIDerivedType(tag: DW_TAG_member, name: "flags", scope: !176, file: !5, baseType: !26, size: 32, align: 32, flags: DIFlagPrivate)
!179 = !DIDerivedType(tag: DW_TAG_member, name: "width", scope: !176, file: !5, baseType: !110, size: 16, align: 16, offset: 32, flags: DIFlagPrivate)
!180 = !DIDerivedType(tag: DW_TAG_member, name: "precision", scope: !176, file: !5, baseType: !110, size: 16, align: 16, offset: 48, flags: DIFlagPrivate)
!181 = !DIDerivedType(tag: DW_TAG_member, name: "buf", scope: !173, file: !5, baseType: !182, size: 128, align: 64, flags: DIFlagPrivate)
!182 = !DICompositeType(tag: DW_TAG_structure_type, name: "&mut dyn core::fmt::Write", file: !5, size: 128, align: 64, elements: !183, templateParams: !27, identifier: "5d81b31fae83b726a211515f989435e3")
!183 = !{!184, !187}
!184 = !DIDerivedType(tag: DW_TAG_member, name: "pointer", scope: !182, file: !5, baseType: !185, size: 64, align: 64)
!185 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !186, size: 64, align: 64, dwarfAddressSpace: 0)
!186 = !DICompositeType(tag: DW_TAG_structure_type, name: "dyn core::fmt::Write", file: !5, align: 8, elements: !27, identifier: "81edc0542d22de6e6d12b7d0c49b319e")
!187 = !DIDerivedType(tag: DW_TAG_member, name: "vtable", scope: !182, file: !5, baseType: !188, size: 64, align: 64, offset: 64)
!188 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&[usize; 6]", baseType: !189, size: 64, align: 64, dwarfAddressSpace: 0)
!189 = !DICompositeType(tag: DW_TAG_array_type, baseType: !80, size: 384, align: 64, elements: !190)
!190 = !{!191}
!191 = !DISubrange(count: 6, lowerBound: 0)
!192 = !DIDerivedType(tag: DW_TAG_member, name: "_lifetime", scope: !139, file: !5, baseType: !193, align: 8, offset: 128, flags: DIFlagPrivate)
!193 = !DICompositeType(tag: DW_TAG_structure_type, name: "PhantomData<&()>", scope: !194, file: !5, align: 8, flags: DIFlagPublic, elements: !27, templateParams: !195, identifier: "17a1071645e7fcb65a0cca5b7ffec91f")
!194 = !DINamespace(name: "marker", scope: !10)
!195 = !{!196}
!196 = !DITemplateTypeParameter(name: "T", type: !197)
!197 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&()", baseType: !148, size: 64, align: 64, dwarfAddressSpace: 0)
!198 = !DIDerivedType(tag: DW_TAG_member, name: "Count", scope: !136, file: !5, baseType: !199, size: 128, align: 64, extraData: i64 0)
!199 = !DICompositeType(tag: DW_TAG_structure_type, name: "Count", scope: !134, file: !5, size: 128, align: 64, flags: DIFlagPrivate, elements: !200, templateParams: !27, identifier: "5fb8095a7fe2cca59d31f4e9db75cf48")
!200 = !{!201}
!201 = !DIDerivedType(tag: DW_TAG_member, name: "__0", scope: !199, file: !5, baseType: !110, size: 16, align: 16, offset: 64, flags: DIFlagPrivate)
!202 = !DIDerivedType(tag: DW_TAG_member, scope: !134, file: !5, baseType: !125, size: 64, align: 64, flags: DIFlagArtificial)
!203 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !127, file: !5, baseType: !80, size: 64, align: 64, offset: 64)
!204 = !DIDerivedType(tag: DW_TAG_member, name: "location", scope: !60, file: !5, baseType: !205, size: 64, align: 64, offset: 64, flags: DIFlagPrivate)
!205 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "&core::panic::location::Location", baseType: !206, size: 64, align: 64, dwarfAddressSpace: 0)
!206 = !DICompositeType(tag: DW_TAG_structure_type, name: "Location", scope: !207, file: !5, size: 192, align: 64, flags: DIFlagPublic, elements: !208, templateParams: !27, identifier: "5463cef928d91281a8d8fb4993bf7cc")
!207 = !DINamespace(name: "location", scope: !62)
!208 = !{!209, !214, !215}
!209 = !DIDerivedType(tag: DW_TAG_member, name: "file_bytes_with_nul", scope: !206, file: !5, baseType: !210, size: 128, align: 64, flags: DIFlagPrivate)
!210 = !DICompositeType(tag: DW_TAG_structure_type, name: "&[u8]", file: !5, size: 128, align: 64, elements: !211, templateParams: !27, identifier: "31681e0c10b314f1f33e38b2779acbb4")
!211 = !{!212, !213}
!212 = !DIDerivedType(tag: DW_TAG_member, name: "data_ptr", scope: !210, file: !5, baseType: !77, size: 64, align: 64)
!213 = !DIDerivedType(tag: DW_TAG_member, name: "length", scope: !210, file: !5, baseType: !80, size: 64, align: 64, offset: 64)
!214 = !DIDerivedType(tag: DW_TAG_member, name: "line", scope: !206, file: !5, baseType: !26, size: 32, align: 32, offset: 128, flags: DIFlagPrivate)
!215 = !DIDerivedType(tag: DW_TAG_member, name: "col", scope: !206, file: !5, baseType: !26, size: 32, align: 32, offset: 160, flags: DIFlagPrivate)
!216 = !DIDerivedType(tag: DW_TAG_member, name: "can_unwind", scope: !60, file: !5, baseType: !217, size: 8, align: 8, offset: 128, flags: DIFlagPrivate)
!217 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!218 = !DIDerivedType(tag: DW_TAG_member, name: "force_no_backtrace", scope: !60, file: !5, baseType: !217, size: 8, align: 8, offset: 136, flags: DIFlagPrivate)
!219 = !{!220}
!220 = !DILocalVariable(name: "_info", arg: 1, scope: !56, file: !3, line: 48, type: !59)
!221 = !DILocation(line: 0, scope: !56)
!222 = !DILocation(line: 49, column: 5, scope: !56)
