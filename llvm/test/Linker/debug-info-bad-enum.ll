; Check that ODR type uniquing does not create invalid debug metadata.
; RUN: split-file %s %t
; RUN: llvm-link -S -o - %t/a.ll %t/b.ll 2>&1 | FileCheck %s

; CHECK-DAG: distinct !DICompileUnit({{.*}}, enums: ![[ENUMLIST:[0-9]+]],
; CHECK-DAG: ![[ENUMLIST]] = !{![[ENUM:[0-9]+]]}
; CHECK-DAG: ![[ENUM]] = distinct !DICompositeType(tag: DW_TAG_enumeration_type, {{.*}}, identifier: "_ZTS5Stage"

; CHECK-DAG: distinct !DICompileUnit({{.*}}, retainedTypes: ![[RETAIN:[0-9]+]],
; CHECK-DAG: ![[RETAIN]] = !{![[VAR:[0-9]+]]}
; CHECK-DAG: ![[CLASS:[0-9]+]] = distinct !DICompositeType(tag: DW_TAG_class_type, {{.*}}, identifier: "_ZTS5Stage"
; CHECK-DAG: ![[VAR]] = !DIDerivedType(tag: DW_TAG_member, name: "Var", scope: ![[CLASS]],

; CHECK-NOT: enum type is not a scope; check enum type ODR violation

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "a.cpp", directory: "file")
!2 = !{!3}
!3 = !DICompositeType(tag: DW_TAG_enumeration_type, name: "Stage", file: !1, line: 3, baseType: !4, size: 32, elements: !5, identifier: "_ZTS5Stage")
!4 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!5 = !{!6}
!6 = !DIEnumerator(name: "A1", value: 0, isUnsigned: true)
!9 = !{i32 2, !"Debug Info Version", i32 3}


;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11}

!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", isOptimized: true, emissionKind: FullDebug, retainedTypes: !10, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "b.cpp", directory: "file")
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_member, name: "Var", scope: !8, file: !3, line: 5, baseType: !5, flags: DIFlagStaticMember)
!8 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "Stage", file: !3, line: 3, size: 64, flags: DIFlagTypePassByValue, elements: !9, identifier: "_ZTS5Stage")
!9 = !{!6}
!10 = !{!6}
!11 = !{i32 2, !"Debug Info Version", i32 3}
