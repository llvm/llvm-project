; RUN: llc -mtriple=x86_64-apple-macosx10.7 %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; CHECK: ptr
; CHECK-NOT: AT_bit_size

source_filename = "test/DebugInfo/X86/pointer-type-size.ll"

%struct.crass = type { i8* }

@crass = common global %struct.crass zeroinitializer, align 8, !dbg !0

!llvm.dbg.cu = !{!9}
!llvm.module.flags = !{!12}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "crass", scope: null, file: !2, line: 1, type: !3, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "foo.c", directory: "/Users/echristo/tmp")
!3 = !DICompositeType(tag: DW_TAG_structure_type, name: "crass", file: !2, line: 1, size: 64, align: 64, elements: !4)
!4 = !{!5}
!5 = !DIDerivedType(tag: DW_TAG_member, name: "ptr", scope: !3, file: !2, line: 1, baseType: !6, size: 64, align: 64)
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64)
!8 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!9 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, producer: "clang version 3.1 (trunk 147882)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !10, retainedTypes: !10, globals: !11, imports: !10)
!10 = !{}
!11 = !{!0}
!12 = !{i32 1, !"Debug Info Version", i32 3}

