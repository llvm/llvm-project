; REQUIRES: x86-registered-target
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

%struct.t1 = type { i32 }

@g1 = dso_local global %struct.t1 zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g1", scope: !2, file: !3, line: 7, type: !6, isLocal: false, isDefinition: true, annotations: !10)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 47af5574a87dc298b5c6c36ff6a969c8c77c8499)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/work/tests/llvm/btf_tag")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t1", file: !3, line: 4, size: 32, elements: !7)
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !6, file: !3, line: 5, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11, !12}
!11 = !{!"btf_decl_tag", !"tag1"}
!12 = !{!"btf_decl_tag", !"tag2"}

; CHECK:        distinct !DIGlobalVariable(name: "g1"
; CHECK-SAME:   annotations: ![[ANNOT:[0-9]+]]
; CHECK:        ![[ANNOT]] = !{![[TAG1:[0-9]+]], ![[TAG2:[0-9]+]]}
; CHECK:        ![[TAG1]] = !{!"btf_decl_tag", !"tag1"}
; CHECK:        ![[TAG2]] = !{!"btf_decl_tag", !"tag2"}

!13 = !{i32 7, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 7, !"uwtable", i32 1}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 47af5574a87dc298b5c6c36ff6a969c8c77c8499)"}
