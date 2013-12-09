; RUN: llc -mtriple=x86_64-pc-linux-gnu -generate-gnu-dwarf-pub-sections -filetype=obj < %s | llvm-dwarfdump - | FileCheck %s

; Generated from:

; static int a __attribute__((section("a")));

; Check that the attributes in the compile unit both point to a correct
; location, even when nothing is exported.
; CHECK: DW_AT_GNU_pubnames [DW_FORM_sec_offset]   (0x00000000)
; CHECK: DW_AT_GNU_pubtypes [DW_FORM_sec_offset]   (0x00000000)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.4 (trunk 191846) (llvm/trunk 191866)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/foo.c] [DW_LANG_C99]
!1 = metadata !{metadata !"foo.c", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!4 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
