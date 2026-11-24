; Base types synthesized for DW_OP_convert are named DW_TAG_base_type DIEs, so
; they must be indexed in the accelerator tables like any other named type.
; Otherwise `llvm-dwarfdump --verify` reports them as missing from .debug_names.

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -O0 \
; RUN:   -dwarf-op-convert=Enable %s -o %t.o
; RUN: llvm-dwarfdump --verify %t.o | FileCheck %s --check-prefix=VERIFY
; RUN: llvm-dwarfdump --find=DW_ATE_signed_8 --find=DW_ATE_signed_32 \
; RUN:   --find=int %t.o | FileCheck %s --check-prefix=FIND

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -O0 \
; RUN:   -dwarf-op-convert=Enable -split-dwarf-file=%t.dwo \
; RUN:   -split-dwarf-output=%t.dwo %s -o %t.split.o
; RUN: llvm-dwarfdump --verify %t.split.o | FileCheck %s --check-prefix=VERIFY

; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj -O0 \
; RUN:   -dwarf-op-convert=Enable -accel-tables=Apple %s -o %t.apple.o
; RUN: llvm-dwarfdump --verify %t.apple.o | FileCheck %s --check-prefix=VERIFY
; RUN: llvm-dwarfdump --find=DW_ATE_signed_8 --find=DW_ATE_signed_32 \
; RUN:   --find=int %t.apple.o | FileCheck %s --check-prefix=FIND

; VERIFY: No errors.

; FIND-DAG: DW_AT_name ("DW_ATE_signed_8")
; FIND-DAG: DW_AT_name ("DW_ATE_signed_32")
; FIND-DAG: DW_AT_name ("int")

; 'int' is a source-level type; DW_ATE_signed_8 and DW_ATE_signed_32 are
; synthesized for the DW_OP_LLVM_convert operations. The non-convert operation
; keeps the conversions from being folded away.
define dso_local i32 @main() !dbg !4 {
  #dbg_value(i8 -42, !9, !DIExpression(DW_OP_lit0, DW_OP_plus, DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_LLVM_convert, 32, DW_ATE_signed, DW_OP_stack_value), !10)
  ret i32 0, !dbg !11
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: Default)
!1 = !DIFile(filename: "convert-debug-names.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILocalVariable(name: "converted", scope: !4, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 2, column: 7, scope: !4)
!11 = !DILocation(line: 3, column: 3, scope: !4)
