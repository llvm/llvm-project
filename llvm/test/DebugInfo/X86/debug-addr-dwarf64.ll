; This checks that .debug_addr can be generated in the DWARF64 format.

; RUN: llc -mtriple=x86_64 -dwarf-version=5 -dwarf64 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-info -debug-addr %t | FileCheck %s

; CHECK:      .debug_info contents:
; CHECK:      DW_TAG_compile_unit
; CHECK:        DW_AT_addr_base (0x0000000000000010)

; CHECK:      .debug_addr contents:
; CHECK-NEXT: Address table header: length = 0x0000000000000014, format = DWARF64, version = 0x0005, addr_size = 0x08, seg_size = 0x00
; CHECK-NEXT: Addrs: [
; CHECK-NEXT: 0x0000000000000000
; CHECK-NEXT: 0x0000000000000004
; CHECK-NEXT: ]

; IR generated and reduced from:
; $ cat foo.c
; int foo;
; int bar;
; $ clang -g -S -emit-llvm foo.c -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

@foo = dso_local global i32 0, align 4, !dbg !0
@bar = dso_local global i32 0, align 4, !dbg !6

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "foo.c", directory: "/tmp")
!4 = !{}
!5 = !{!0, !6}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "bar", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{!"clang version 12.0.0"}
