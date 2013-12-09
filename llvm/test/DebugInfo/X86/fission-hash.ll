; RUN: llc -split-dwarf=Enable -generate-cu-hash -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s

; The source is an empty file.

; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0x0c1e629c9e5ada4f)
; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0x0c1e629c9e5ada4f)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.4 (trunk 188230) (llvm/trunk 188234)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !2, metadata !2, metadata !2, metadata !"foo.dwo"} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/foo.c] [DW_LANG_C99]
!1 = metadata !{metadata !"foo.c", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!4 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
