; RUN: llc -mtriple=x86_64-linux-gnu -O0 -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s

; Generated from:
; clang -g -S -emit-llvm -o foo.ll foo.c
; typedef int v4si __attribute__((__vector_size__(16)));
;
; v4si a

source_filename = "test/DebugInfo/X86/vector.ll"

@a = common global <4 x i32> zeroinitializer, align 16, !dbg !0

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "a", scope: null, file: !2, line: 3, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "foo.c", directory: "/Users/echristo")
!3 = !DIDerivedType(tag: DW_TAG_typedef, name: "v4si", file: !2, line: 1, baseType: !4)
!4 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 128, align: 128, flags: DIFlagVector, elements: !6)
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{!7}
!7 = !DISubrange(count: 4)
!8 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.3 (trunk 171825) (llvm/trunk 171822)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !9, retainedTypes: !9, globals: !10, imports: !9)
!9 = !{}
; Check that we get an array type with a vector attribute.
; CHECK: DW_TAG_array_type
; CHECK-NEXT: DW_AT_GNU_vector
!10 = !{!0}
!11 = !{i32 1, !"Debug Info Version", i32 3}
