; Ensures that the AsmPrinter rounds up zero-sized symbols in `.debug_aranges` to one byte.

; Generated from the following Rust source:
;
;     pub static EXAMPLE: () = ();
;
; Compiled with:
;
;     $ rustc --crate-type=lib --target=x86_64-unknown-linux-gnu --emit=llvm-ir -g dwarf-aranges-zero-size.rs

; RUN: llc --generate-arange-section < %s | FileCheck %s
; CHECK: .section .debug_aranges
; CHECK: .quad _ZN23dwarf_aranges_zero_size7EXAMPLE17h8ab19f2b0c3b238dE
; CHECK-NEXT: .quad 1
; CHECK: .section

; ModuleID = 'dwarf_aranges_zero_size.fbc28187-cgu.0'
source_filename = "dwarf_aranges_zero_size.fbc28187-cgu.0"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZN23dwarf_aranges_zero_size7EXAMPLE17h8ab19f2b0c3b238dE = constant <{ [0 x i8] }> zeroinitializer, align 1, !dbg !0
@__rustc_debug_gdb_scripts_section__ = linkonce_odr unnamed_addr constant [34 x i8] c"\01gdb_load_rust_pretty_printers.py\00", section ".debug_gdb_scripts", align 1

!llvm.module.flags = !{!5, !6, !7}
!llvm.dbg.cu = !{!8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "EXAMPLE", linkageName: "_ZN23dwarf_aranges_zero_size7EXAMPLE17h8ab19f2b0c3b238dE", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true, align: 8)
!2 = !DINamespace(name: "dwarf_aranges_zero_size", scope: null)
!3 = !DIFile(filename: "dwarf-aranges-zero-size.rs", directory: "/Users/pcwalton/Desktop", checksumkind: CSK_MD5, checksum: "c2d51547bfaf5562b9c9061311fe4140")
!4 = !DIBasicType(name: "()", encoding: DW_ATE_unsigned)
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 2, !"RtLibUseGOT", i32 1}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !9, producer: "clang LLVM (rustc version 1.60.0-nightly (0c292c966 2022-02-08))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !10, globals: !11)
!9 = !DIFile(filename: "dwarf-aranges-zero-size.rs/@/dwarf_aranges_zero_size.fbc28187-cgu.0", directory: "/Users/pcwalton/Desktop")
!10 = !{}
!11 = !{!0}
