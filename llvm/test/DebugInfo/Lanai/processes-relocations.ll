; RUN: llvm-dwarfdump %p/Inputs/lanai-processes-relocations.elf 2>&1 | FileCheck %s

; FIXME: Use llc with this file as input instead of binary file.
; NOTE: this test is currently not using llc, but using a binary input as the
; rest of the backend is not yet in tree. Once the Lanai backend is in tree,
; the binary file will be removed and this test will use llc.

; CHECK-NOT: failed to compute relocation

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = !{i32 786449, !1, i32 12, !"clang version 3.6.0 ", i1 false, !"", i32 0, !2, !2, !2, !2, !2, !"", i32 1} ; [ DW_TAG_compile_unit ] [/a/empty.c] [DW_LANG_C99]
!1 = !{!"empty.c", !"/a"}
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.6.0 "}
