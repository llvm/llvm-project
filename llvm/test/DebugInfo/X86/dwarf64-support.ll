; This checks cases when the 64-bit DWARF debug info should not be generated
; even if '-dwarf64' is specified.

; The 64-bit DWARF format was introduced in DWARFv3, so the '-dwarf64' switch
; should be ignored for earlier versions.
; RUN: llc -mtriple=x86_64 -dwarf-version=2 -dwarf64 -filetype=obj %s -o - | \
; RUN:   llvm-dwarfdump -debug-line - | \
; RUN:   FileCheck %s --check-prefixes=ELF64,CHECK

; DWARF64 requires 64-bit relocations, so it is not produced for 32-bit targets.
; RUN: llc -mtriple=i386 -dwarf-version=5 -dwarf64 -filetype=obj %s -o - | \
; RUN:   llvm-dwarfdump -debug-line - | \
; RUN:   FileCheck %s --check-prefixes=ELF32,CHECK

; DWARF64 is enabled only for ELF targets. The switch should be ignored for COFF.
; RUN: llc -mtriple=x86_64-windows-gnu -dwarf-version=5 -dwarf64 -filetype=obj %s -o - | \
; RUN:   llvm-dwarfdump -debug-line - | \
; RUN:   FileCheck %s --check-prefixes=COFF,CHECK

; DWARF64 is enabled only for ELF targets. The switch should be ignored for Mach-O.
; RUN: llc -mtriple=x86_64-apple-darwin -dwarf-version=5 -dwarf64 -filetype=obj %s -o - | \
; RUN:   llvm-dwarfdump -debug-line - | \
; RUN:   FileCheck %s --check-prefixes=MACHO,CHECK

; ELF64:    file format elf64-x86-64
; ELF32:    file format elf32-i386
; COFF:     file format COFF-x86-64
; MACHO:    file format Mach-O 64-bit x86-64

; CHECK:      .debug_line contents:
; CHECK-NEXT: debug_line[0x00000000]
; CHECK-NEXT: Line table prologue:
; CHECK-NEXT:     total_length:
; CHECK-NEXT:         format: DWARF32

; IR generated and reduced from:
; $ cat foo.c
; int foo;
; $ clang -g -S -emit-llvm foo.c -o foo.ll

target triple = "x86_64-unknown-linux-gnu"

@foo = dso_local global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "foo", scope: !2, file: !3, line: 1, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "foo.c", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{!"clang version 12.0.0"}
