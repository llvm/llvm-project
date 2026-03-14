; RUN: llvm-as < %s | llvm-dis | FileCheck %s
;
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   int __tag1 *g;
; Compilation flag:
;   clang -S -g -emit-llvm test.c

@g = dso_local global i32* null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!9, !10, !11, !12, !13}
!llvm.ident = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git 248122328bfefe82608a2e110af3a3ff04279ddf)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/btf_tag_type")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, annotations: !7)
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !{!8}
!8 = !{!"btf_type_tag", !"tag1"}

; CHECK:       distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 2, type: !5
; CHECK:       !5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, annotations: !7)
; CHECK-NEXT:  !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
; CHECK-NEXT:  !7 = !{!8}
; CHECK-NEXT:  !8 = !{!"btf_type_tag", !"tag1"}

!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 7, !"uwtable", i32 1}
!13 = !{i32 7, !"frame-pointer", i32 2}
!14 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git 248122328bfefe82608a2e110af3a3ff04279ddf)"}
