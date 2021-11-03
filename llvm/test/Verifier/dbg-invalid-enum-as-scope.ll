; RUN: llvm-as -disable-output <%s 2>&1 | FileCheck %s
; CHECK: enum type is not a scope; check enum type ODR violation
; CHECK: warning: ignoring invalid debug info

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}
!llvm.dbg.cu = !{!1}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !13, enums: !3)
!2 = !DIFile(filename: "file.c", directory: "dir")
!3 = !{!4}
!4 = distinct !DICompositeType(tag: DW_TAG_enumeration_type, name: "Stage", file: !2, line: 3, baseType: !10, size: 32, elements: !11, identifier: "_ZTS5Stage")
!6 = !DIDerivedType(tag: DW_TAG_member, name: "Var", scope: !4, file: !2, line: 5, baseType: !10)
!10 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!11 = !{!12}
!12 = !DIEnumerator(name: "A1", value: 0, isUnsigned: true)
!13 = !{!6}
