; UNSUPPORTED: target={{.*}}-aix{{.*}}
; REQUIRES: object-emission
; RUN: llc -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; Source:
;   #define __tag1 __attribute__((btf_type_tag("tag1")))
;   #define __tag2 __attribute__((btf_type_tag("tag2")))
;
;   int * __tag1 * __tag2 *g;
; Compilation flag:
;   clang -target x86_64 -g -S -emit-llvm t.c

@g = dso_local global ptr null, align 8, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!13, !14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g", scope: !2, file: !3, line: 4, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git 2c240a5eefae1a945dfd36cdaa0c677eca90dd82)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.c", directory: "/home/yhs/work/tests/llvm/btf_tag_type")
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64, annotations: !11)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, annotations: !9)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !{!"btf_type_tag", !"tag1"}
!11 = !{!12}
!12 = !{!"btf_type_tag", !"tag2"}

; CHECK:      DW_TAG_variable
; CHECK-NEXT:   DW_AT_name      ("g")
; CHECK-NEXT:   DW_AT_type      (0x[[T1:[0-9a-f]+]] "int ***")

; CHECK:      0x[[T1]]: DW_TAG_pointer_type
; CHECK-NEXT:   DW_AT_type      (0x[[T2:[0-9a-f]+]] "int **")

; CHECK:        DW_TAG_LLVM_annotation
; CHECK-NEXT:     DW_AT_name    ("btf_type_tag")
; CHECK-NEXT:     DW_AT_const_value     ("tag2")

; CHECK:        NULL

; CHECK:      0x[[T2]]: DW_TAG_pointer_type
; CHECK-NEXT:   DW_AT_type      (0x[[T3:[0-9a-f]+]] "int *")

; CHECK:        DW_TAG_LLVM_annotation
; CHECK-NEXT:     DW_AT_name    ("btf_type_tag")
; CHECK-NEXT:     DW_AT_const_value     ("tag1")

; CHECK:        NULL

; CHECK:      0x[[T3]]: DW_TAG_pointer_type
; CHECK-NEXT:   DW_AT_type      (0x{{[0-9a-f]+}} "int")

!13 = !{i32 7, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 7, !"uwtable", i32 1}
!17 = !{i32 7, !"frame-pointer", i32 2}
!18 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git 2c240a5eefae1a945dfd36cdaa0c677eca90dd82)"}
