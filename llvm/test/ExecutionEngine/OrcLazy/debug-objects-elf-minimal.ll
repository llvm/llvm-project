; REQUIRES: native && target-x86_64

; In-memory debug-object contains some basic DWARF
;
; RUN: lli --jit-linker=rtdyld \
; RUN:     --generate=__dump_jit_debug_objects %s | llvm-dwarfdump --diff - | FileCheck %s
;
; RUN: lli --jit-linker=jitlink \
; RUN:     --generate=__dump_jit_debug_objects %s | llvm-dwarfdump --diff - | FileCheck %s
;
; CHECK: -:	file format elf64-x86-64
; CHECK: .debug_info contents:
; CHECK: 0x00000000: Compile Unit: length = 0x00000047, format = DWARF32, version = 0x0004, abbr_offset = 0x0000, addr_size = 0x08 (next unit at 0x0000004b)
; CHECK: DW_TAG_compile_unit
; CHECK:               DW_AT_producer	("compiler version")
; CHECK:               DW_AT_language	(DW_LANG_C99)
; CHECK:               DW_AT_name	("source-file.c")
; CHECK:               DW_AT_stmt_list	()
; CHECK:               DW_AT_comp_dir	("/workspace")
; CHECK:               DW_AT_low_pc	()
; CHECK:               DW_AT_high_pc	()
; CHECK:   DW_TAG_subprogram
; CHECK:                 DW_AT_low_pc	()
; CHECK:                 DW_AT_high_pc	()
; CHECK:                 DW_AT_frame_base	(DW_OP_reg7 RSP)
; CHECK:                 DW_AT_name	("main")
; CHECK:                 DW_AT_decl_file	("/workspace/source-file.c")
; CHECK:                 DW_AT_decl_line	(4)
; CHECK:                 DW_AT_type	("int")
; CHECK:                 DW_AT_external	(true)
; CHECK:   DW_TAG_base_type
; CHECK:                 DW_AT_name	("int")
; CHECK:                 DW_AT_encoding	(DW_ATE_signed)
; CHECK:                 DW_AT_byte_size	(0x04)
; CHECK:   NULL

; Text section of the in-memory debug-object has a non-null load-address
;
; RUN: lli --jit-linker=rtdyld \
; RUN:     --generate=__dump_jit_debug_objects %s | llvm-objdump --section-headers - | \
; RUN:     FileCheck --check-prefix=CHECK_LOAD_ADDR %s
;
; RUN: lli --jit-linker=jitlink \
; RUN:     --generate=__dump_jit_debug_objects %s | llvm-objdump --section-headers - | \
; RUN:     FileCheck --check-prefix=CHECK_LOAD_ADDR %s
;
; CHECK_LOAD_ADDR-NOT: {{[0-9]*}} .text {{.*}} 0000000000000000 TEXT

target triple = "x86_64-unknown-unknown-elf"

; Built-in symbol provided by the JIT
declare void @__dump_jit_debug_objects(i8*)

; Host-process symbol from the GDB JIT interface
@__jit_debug_descriptor = external global i8, align 1

define i32 @main() !dbg !9 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @__dump_jit_debug_objects(i8* @__jit_debug_descriptor), !dbg !13
  ret i32 0, !dbg !14
}

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.dbg.cu = !{!5}
!llvm.ident = !{!8}

!0 = !{i32 2, !"SDK Version", [3 x i32] [i32 10, i32 15, i32 6]}
!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"PIC Level", i32 2}
!5 = distinct !DICompileUnit(language: DW_LANG_C99, file: !6, producer: "compiler version", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, nameTableKind: None)
!6 = !DIFile(filename: "source-file.c", directory: "/workspace")
!7 = !{}
!8 = !{!"compiler version"}
!9 = distinct !DISubprogram(name: "main", scope: !6, file: !6, line: 4, type: !10, scopeLine: 4, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !7)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 5, column: 3, scope: !9)
!14 = !DILocation(line: 6, column: 3, scope: !9)
