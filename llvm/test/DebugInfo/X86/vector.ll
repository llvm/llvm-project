; RUN: llc -mtriple=x86_64-linux-gnu -O0 -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; Generated from:
; clang -g -S -emit-llvm -o foo.ll foo.c
; typedef int v4si __attribute__((__vector_size__(16)));
;
; v4si a

@a = common global <4 x i32> zeroinitializer, align 16

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.3 (trunk 171825) (llvm/trunk 171822)", isOptimized: false, emissionKind: FullDebug, file: !12, enums: !1, retainedTypes: !1, subprograms: !1, globals: !3, imports:  !1)
!1 = !{}
!3 = !{!5}
!5 = !DIGlobalVariable(name: "a", line: 3, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7, variable: <4 x i32>* @a)
!6 = !DIFile(filename: "foo.c", directory: "/Users/echristo")
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "v4si", line: 1, file: !12, baseType: !8)
!8 = !DICompositeType(tag: DW_TAG_array_type, size: 128, align: 128, flags: DIFlagVector, baseType: !9, elements: !10)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DISubrange(count: 4)
!12 = !DIFile(filename: "foo.c", directory: "/Users/echristo")

; Check that we get an array type with a vector attribute.
; CHECK: DW_TAG_array_type
; CHECK-NEXT: DW_AT_GNU_vector
!13 = !{i32 1, !"Debug Info Version", i32 3}
