; RUN: llc -split-dwarf-file=foo.dwo -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump -v -all %t | FileCheck %s

; The source is an empty file, modified to include/retain an 'int' type, since empty CUs are omitted.

; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0xb1b50e9a23896bc1)
; CHECK: DW_AT_GNU_dwo_id [DW_FORM_data8] (0xb1b50e9a23896bc1)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.4 (trunk 188230) (llvm/trunk 188234)", isOptimized: false, splitDebugFilename: "foo.dwo", emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !5, globals: !2, imports: !2)
!1 = !DIFile(filename: "foo.c", directory: "/usr/local/google/home/echristo/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 3}
!4 = !{i32 1, !"Debug Info Version", i32 3}
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
