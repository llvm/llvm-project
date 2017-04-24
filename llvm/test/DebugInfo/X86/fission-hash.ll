; RUN: llc -split-dwarf-file=foo.dwo -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-dump=all %t | FileCheck %s

; The source is an empty file.

; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0x0c1e629c9e5ada4f)
; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0x0c1e629c9e5ada4f)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 (trunk 188230) (llvm/trunk 188234)", isOptimized: false, splitDebugFilename: "foo.dwo", emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "foo.c", directory: "/usr/local/google/home/echristo/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 3}
!4 = !{i32 1, !"Debug Info Version", i32 3}
