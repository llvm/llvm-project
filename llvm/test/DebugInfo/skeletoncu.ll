; RUN: %llc_dwarf %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_GNU_dwo_id {{.*}}abcd
; CHECK: DW_AT_GNU_dwo_name {{.*}}"my.dwo"
; REQUIRES: object-emission
 
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "LLVM", isOptimized: false, runtimeVersion: 2, splitDebugFilename: "my.dwo", emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2, dwoId: 43981)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
