; RUN: llc < %s | FileCheck %s

; CHECK-NOT: S_GDATA32

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

@"\01?id@?$numpunct@D@@0HA" = available_externally dllimport global i32 0, align 4

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 272628) (llvm/trunk 272566)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "/usr/local/google/home/majnemer/Downloads/<stdin>", directory: "/usr/local/google/home/majnemer/llvm/src")
!2 = !{}
!3 = !{!4}
!4 = distinct !DIGlobalVariable(name: "id", linkageName: "\01?id@?$numpunct@D@@0HA", scope: !0, file: !5, line: 4, type: !6, isLocal: false, isDefinition: true, variable: i32* @"\01?id@?$numpunct@D@@0HA", declaration: !7)
!5 = !DIFile(filename: "/usr/local/google/home/majnemer/Downloads/t.ii", directory: "/usr/local/google/home/majnemer/llvm/src")
!6 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "id", scope: !8, file: !5, line: 2, baseType: !6, flags: DIFlagStaticMember)
!8 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "numpunct<char>", file: !5, line: 2, size: 8, align: 8, elements: !9, templateParams: !10)
!9 = !{!7}
!10 = !{!11}
!11 = !DITemplateTypeParameter(type: !12)
!12 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!13 = !{i32 2, !"CodeView", i32 1}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.9.0 (trunk 272628) (llvm/trunk 272566)"}
