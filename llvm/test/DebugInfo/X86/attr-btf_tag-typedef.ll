; RUN: llc -mtriple=x86_64-linux -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
;
; Source:
;   #define __tag1 __attribute__((btf_decl_tag("tag1")))
;   typedef struct { int a; } __s __tag1;
;   typedef unsigned * __u __tag1;
;   __s a;
;   __u u;
; Compilation flag:
;   clang -S -g -emit-llvm typedef.c

%struct.__s = type { i32 }

@a = dso_local global %struct.__s zeroinitializer, align 4, !dbg !0
@u = dso_local global ptr null, align 8, !dbg !5

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18, !19, !20, !21}
!llvm.ident = !{!22}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 4, type: !12, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git b9757992b73e823edf1fa699372ff9cd29db6cb7)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "typedef.c", directory: "/home/yhs/work/tests/llvm/btf_tag")
!4 = !{!0, !5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "u", scope: !2, file: !3, line: 5, type: !7, isLocal: false, isDefinition: true)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "__u", file: !3, line: 3, baseType: !8, annotations: !10)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64)
!9 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!10 = !{!11}
!11 = !{!"btf_decl_tag", !"tag1"}
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "__s", file: !3, line: 2, baseType: !13, annotations: !10)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, file: !3, line: 2, size: 32, elements: !14)
!14 = !{!15}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !3, line: 2, baseType: !16, size: 32)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

; CHECK:        DW_TAG_typedef
; CHECK-NEXT:     DW_AT_type
; CHECK-NEXT:     DW_AT_name      ("__s")
; CHECK-NEXT:     DW_AT_decl_file
; CHECK-NEXT:     DW_AT_decl_line
; CHECK-EMPTY:
; CHECK-NEXT:     DW_TAG_LLVM_annotation
; CHECK-NEXT:       DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:       DW_AT_const_value     ("tag1")
; CHECK-EMPTY:
; CHECK-NEXT:     NULL

; CHECK:        DW_TAG_typedef
; CHECK-NEXT:     DW_AT_type
; CHECK-NEXT:     DW_AT_name      ("__u")
; CHECK-NEXT:     DW_AT_decl_file
; CHECK-NEXT:     DW_AT_decl_line
; CHECK-EMPTY:
; CHECK-NEXT:     DW_TAG_LLVM_annotation
; CHECK-NEXT:       DW_AT_name    ("btf_decl_tag")
; CHECK-NEXT:       DW_AT_const_value     ("tag1")
; CHECK-EMPTY:
; CHECK-NEXT:     NULL

!17 = !{i32 7, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{i32 7, !"uwtable", i32 1}
!21 = !{i32 7, !"frame-pointer", i32 2}
!22 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git b9757992b73e823edf1fa699372ff9cd29db6cb7)"}
