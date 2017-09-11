; RUN: %llc_dwarf %s -filetype=obj -o %t
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; REQUIRES: default_triple
;
; CHECK: DW_TAG_compile_unit
; CHECK-NOT: dwo_id
;
; The skeleton must come second or LLDB may get confused.
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_GNU_dwo_id {{.*}}abcd
; CHECK: DW_AT_GNU_dwo_name {{.*}}"my.dwo"

!llvm.dbg.cu = !{!7, !0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Clang", isOptimized: false, runtimeVersion: 2, splitDebugFilename: "my.dwo", emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2, dwoId: 43981)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Clang", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, retainedTypes: !5)
