; RUN: llc -mtriple=x86_64-apple-macosx10.7 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: ptr
; CHECK-NOT: AT_bit_size

%struct.crass = type { i8* }

@crass = common global %struct.crass zeroinitializer, align 8

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.1 (trunk 147882)", isOptimized: false, emissionKind: FullDebug, file: !13, enums: !1, retainedTypes: !1, subprograms: !1, globals: !3, imports:  !1)
!1 = !{}
!3 = !{!5}
!5 = !DIGlobalVariable(name: "crass", line: 1, isLocal: false, isDefinition: true, scope: null, file: !6, type: !7, variable: %struct.crass* @crass)
!6 = !DIFile(filename: "foo.c", directory: "/Users/echristo/tmp")
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "crass", line: 1, size: 64, align: 64, file: !13, elements: !8)
!8 = !{!9}
!9 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", line: 1, size: 64, align: 64, file: !13, scope: !7, baseType: !10)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !12)
!12 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!13 = !DIFile(filename: "foo.c", directory: "/Users/echristo/tmp")
!14 = !{i32 1, !"Debug Info Version", i32 3}
