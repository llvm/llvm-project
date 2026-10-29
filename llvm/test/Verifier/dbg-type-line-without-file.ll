; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK: line specified with no file
; CHECK: warning: ignoring invalid debug info

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, retainedTypes: !3)
!2 = !DIFile(filename: "foo.c", directory: "")
!3 = !{!4}
!4 = !DICompositeType(tag: DW_TAG_array_type, name: "array1", line: 10, size: 128, align: 32, baseType: !5)
!5 = !DIBasicType(name: "int")
