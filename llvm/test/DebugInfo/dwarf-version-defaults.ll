; REQUIRES: x86-registered-target, aarch64-registered-target

; Pin the state DwarfDebug derives from the DWARF version (default version,
; bitfield encoding, macro section, string offsets table, accelerator tables,
; TLS opcode) on a few targets, so that a change to the defaults or to how the
; version is applied is noticed.

; RUN: llc < %s -mtriple=x86_64-linux-gnu | FileCheck %s --check-prefixes=V4,ELF4 \
; RUN:   --implicit-check-not=DW_AT_bit_offset --implicit-check-not=.debug_macro \
; RUN:   --implicit-check-not=.debug_str_offsets --implicit-check-not=.debug_names \
; RUN:   --implicit-check-not=.debug_addr
; RUN: llc < %s -mtriple=aarch64-linux-gnu | FileCheck %s --check-prefixes=V4,ELF4 \
; RUN:   --implicit-check-not=DW_AT_bit_offset --implicit-check-not=.debug_macro \
; RUN:   --implicit-check-not=.debug_str_offsets --implicit-check-not=.debug_names \
; RUN:   --implicit-check-not=.debug_addr
; RUN: llc < %s -mtriple=arm64-apple-darwin | FileCheck %s --check-prefixes=V4,DARWIN \
; RUN:   --implicit-check-not=DW_AT_bit_offset --implicit-check-not=.debug_str_offsets \
; RUN:   --implicit-check-not=.debug_names --implicit-check-not=.debug_addr

; RUN: llc < %s -mtriple=x86_64-linux-gnu -dwarf-version=2 | FileCheck %s \
; RUN:   --check-prefixes=V2,ELF4 --implicit-check-not=DW_AT_data_bit_offset
; RUN: llc < %s -mtriple=aarch64-linux-gnu -dwarf-version=2 | FileCheck %s \
; RUN:   --check-prefixes=V2,ELF4 --implicit-check-not=DW_AT_data_bit_offset
; RUN: llc < %s -mtriple=arm64-apple-darwin -dwarf-version=2 | FileCheck %s \
; RUN:   --check-prefixes=V2,DARWIN --implicit-check-not=DW_AT_data_bit_offset

; RUN: llc < %s -mtriple=x86_64-linux-gnu -dwarf-version=5 | FileCheck %s --check-prefixes=V5
; RUN: llc < %s -mtriple=aarch64-linux-gnu -dwarf-version=5 | FileCheck %s --check-prefixes=V5

; RUN: llc < %s -mtriple=x86_64-linux-gnu -filetype=obj -o %t.gnu.o
; RUN: llvm-dwarfdump --debug-info %t.gnu.o | FileCheck %s --check-prefix=TLS-GNU
; RUN: llc < %s -mtriple=x86_64-linux-gnu -dwarf-version=5 -filetype=obj -o %t.gnu5.o
; RUN: llvm-dwarfdump --debug-info %t.gnu5.o | FileCheck %s --check-prefix=TLS-GNU
; RUN: llc < %s -mtriple=x86_64-linux-gnu -debugger-tune=lldb -filetype=obj -o %t.lldb.o
; RUN: llvm-dwarfdump --debug-info %t.lldb.o | FileCheck %s --check-prefix=TLS-STD
; RUN: llc < %s -mtriple=arm64-apple-darwin -filetype=obj -o %t.darwin.o
; RUN: llvm-dwarfdump --debug-info %t.darwin.o | FileCheck %s --check-prefix=TLS-STD

; V2: 2{{ *[#;/]+ *}}DWARF version number
; V2: DW_AT_bit_offset
; V4: 4{{ *[#;/]+ *}}DWARF version number
; V4: DW_AT_data_bit_offset
; V5: 5{{ *[#;/]+ *}}DWARF version number
; V5: DW_AT_data_bit_offset

; ELF4: .section .debug_macinfo
; DARWIN: __apple_names

; V5: .debug_macro
; V5: .debug_str_offsets
; V5: .debug_addr
; V5: .debug_names

; TLS-GNU: DW_OP_GNU_push_tls_address
; TLS-STD: DW_OP_form_tls_address

%struct.S = type { i8 }

@g = global %struct.S zeroinitializer, !dbg !0
@t = thread_local global i32 0, !dbg !12

define void @f() !dbg !17 {
  ret void, !dbg !18
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!3}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !4, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !4, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !10, macros: !11)
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DIFile(filename: "t.c", directory: "/test")
!5 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !4, line: 1, size: 8, elements: !6)
!6 = !{!7}
!7 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !5, file: !4, line: 2, baseType: !8, size: 3, offset: 1, flags: DIFlagBitField, extraData: i64 0)
!8 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!0, !12}
!11 = !{!15}
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "t", scope: !2, file: !4, line: 3, type: !9, isLocal: false, isDefinition: true)
!14 = !DISubroutineType(types: !16)
!15 = !DIMacro(type: DW_MACINFO_define, line: 1, name: "X", value: "1")
!16 = !{null}
!17 = distinct !DISubprogram(name: "f", scope: !4, file: !4, line: 4, type: !14, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !2)
!18 = !DILocation(line: 4, scope: !17)
