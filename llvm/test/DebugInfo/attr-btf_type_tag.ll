; REQUIRES: object-emission
; RUN: llc -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;   #define __tag3 __attribute__((btf_type_tag("tag3")))
;   #define __tag4 __attribute__((btf_type_tag("tag4")))
;
;   int   __tag1  a;
;   int * __tag2  b;
;   void  __tag3 *c;
;   void (__tag4 *d)(void);
;
;
; Compilation flag:
;   clang -target x86_64 -g -S -emit-llvm t.c
;
; Note: only "btf:type_tag" annotations are checked for brevity.

@a = dso_local global i32 0, align 4, !dbg !0
@b = dso_local global ptr null, align 8, !dbg !5
@c = dso_local global ptr null, align 8, !dbg !11
@d = dso_local global ptr null, align 8, !dbg !19

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!31, !32, !33, !34, !35}
!llvm.ident = !{!36}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 6, type: !28, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git ffde01565bce81795ba0442108742557a9a4562d)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.c", directory: "/home/eddy/work/tmp", checksumkind: CSK_MD5, checksum: "71845c02e58f6b1a8b0162797b4d3f37")
!4 = !{!0, !5, !11, !19}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())

; CHECK:                 DW_TAG_variable
; CHECK-NEXT:               DW_AT_name      ("a")
; CHECK-NEXT:               DW_AT_type      (0x[[T1:[0-9a-f]+]] "int")

; CHECK:      0x[[T1]]:  DW_TAG_base_type
; CHECK-NEXT:               DW_AT_name      ("int")

; CHECK:                    DW_TAG_LLVM_annotation
; CHECK-NEXT:                 DW_AT_name    ("btf:type_tag")
; CHECK-NEXT:                 DW_AT_const_value     ("tag1")

!6 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 7, type: !7, isLocal: false, isDefinition: true)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, annotations: !9)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !{!"btf:type_tag", !"tag2"}
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())

; CHECK:                DW_TAG_variable
; CHECK-NEXT:               DW_AT_name      ("b")
; CHECK-NEXT:               DW_AT_type      (0x[[T2:[0-9a-f]+]] "int *")

; CHECK:      0x[[T2]]:   DW_TAG_pointer_type
; CHECK-NEXT:               DW_AT_type      (0x[[T3:[0-9a-f]+]] "int")

; CHECK:                    DW_TAG_LLVM_annotation
; CHECK-NEXT:                 DW_AT_name    ("btf:type_tag")
; CHECK-NEXT:                 DW_AT_const_value     ("tag2")

; CHECK:      0x[[T3]]:   DW_TAG_base_type
; CHECK-NEXT:               DW_AT_name      ("int")
; CHECK-NEXT:               DW_AT_encoding  (DW_ATE_signed)

!12 = distinct !DIGlobalVariable(name: "c", scope: !2, file: !3, line: 8, type: !13, isLocal: false, isDefinition: true)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 64, annotations: !17)
!14 = !DIBasicType(tag: DW_TAG_unspecified_type, name: "void", annotations: !15)
!15 = !{!16}
!16 = !{!"btf:type_tag", !"tag3"}
!17 = !{!18}
!18 = !{!"btf_type_tag", !"tag3"}

; CHECK:                DW_TAG_variable
; CHECK-NEXT:             DW_AT_name      ("c")
; CHECK-NEXT:             DW_AT_type      (0x[[T4:[0-9a-f]+]] "void *")

; CHECK:      0x[[T4]]: DW_TAG_pointer_type
; CHECK-NEXT:             DW_AT_type      (0x[[T5:[0-9a-f]+]] "void")

; CHECK:      0x[[T5]]: DW_TAG_unspecified_type
; CHECK-NEXT:             DW_AT_name      ("void")

; CHECK:                  DW_TAG_LLVM_annotation
; CHECK-NEXT:               DW_AT_name    ("btf:type_tag")
; CHECK-NEXT:               DW_AT_const_value     ("tag3")

!19 = !DIGlobalVariableExpression(var: !20, expr: !DIExpression())
!20 = distinct !DIGlobalVariable(name: "d", scope: !2, file: !3, line: 9, type: !21, isLocal: false, isDefinition: true)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64, annotations: !26)
!22 = !DISubroutineType(types: !23, annotations: !24)
!23 = !{null}
!24 = !{!25}
!25 = !{!"btf:type_tag", !"tag4"}
!26 = !{!27}
!27 = !{!"btf_type_tag", !"tag4"}
!28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed, annotations: !29)
!29 = !{!30}
!30 = !{!"btf:type_tag", !"tag1"}

; CHECK:                DW_TAG_variable
; CHECK-NEXT:             DW_AT_name      ("d")
; CHECK-NEXT:             DW_AT_type      (0x[[T6:[0-9a-f]+]] "void (*)(")

; CHECK:      0x[[T6]]: DW_TAG_pointer_type
; CHECK-NEXT:             DW_AT_type      (0x[[T7:[0-9a-f]+]] "void (")

; CHECK:      0x[[T7]]: DW_TAG_subroutine_type
; CHECK-NEXT:             DW_AT_prototyped        (true)

; CHECK:                  DW_TAG_LLVM_annotation
; CHECK-NEXT:               DW_AT_name    ("btf:type_tag")
; CHECK-NEXT:               DW_AT_const_value     ("tag4")

!31 = !{i32 7, !"Dwarf Version", i32 5}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 1, !"wchar_size", i32 4}
!34 = !{i32 7, !"uwtable", i32 2}
!35 = !{i32 7, !"frame-pointer", i32 2}
!36 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git ffde01565bce81795ba0442108742557a9a4562d)"}
