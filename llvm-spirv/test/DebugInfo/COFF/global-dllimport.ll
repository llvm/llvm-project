; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=%triple < %t.ll | FileCheck %s

; CHECK-NOT: S_GDATA32

source_filename = "test/DebugInfo/COFF/global-dllimport.ll"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "spir64-unknown-unknown"

@"\01?id@?$numpunct@D@@0HA" = available_externally dllimport global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15}
!llvm.ident = !{!16}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "id", linkageName: "\01?id@?$numpunct@D@@0HA", scope: !2, file: !6, line: 4, type: !7, isLocal: false, isDefinition: true, declaration: !8)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.9.0 (trunk 272628) (llvm/trunk 272566)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "/usr/local/google/home/majnemer/Downloads/<stdin>", directory: "/usr/local/google/home/majnemer/llvm/src")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "/usr/local/google/home/majnemer/Downloads/t.ii", directory: "/usr/local/google/home/majnemer/llvm/src")
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !DIDerivedType(tag: DW_TAG_member, name: "id", scope: !9, file: !6, line: 2, baseType: !7, flags: DIFlagStaticMember)
!9 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "numpunct<char>", file: !6, line: 2, size: 8, align: 8, elements: !10, templateParams: !11)
!10 = !{!8}
!11 = !{!12}
!12 = !DITemplateTypeParameter(type: !13)
!13 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!14 = !{i32 2, !"CodeView", i32 1}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{!"clang version 3.9.0 (trunk 272628) (llvm/trunk 272566)"}

