; REQUIRES: object-emission

; RUN: llc -filetype=obj -O0 -generate-type-units -mtriple=x86_64-unknown-linux-gnu < %s \
; RUN:     | llvm-dwarfdump -debug-info -debug-types - | FileCheck %s

; Test that a type unit referencing a non-type unit (in this case, it's
; bordering on an ODR violation - a type with linkage references a type without
; linkage, so there's no way for the first type to be defined in more than one
; translation unit, so there's no need for it to be in a type unit - but this
; is quirky/rare and an easy way to test a broader issue). The type unit should
; not end up with a whole definition of the referenced type - instead it should
; have a declaration of the type, while the definition remains in the primary
; CU.
; (again, arguably in this instance - since the type is only referenced once, it
; could go in the TU only - but that requires tracking usage & then deciding
; where to put types, which isn't worthwhile right now)

; CHECK: Type Unit:

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"bar"

; CHECK: DW_TAG_namespace
; CHECK-NOT: {{DW_AT_name|DW_TAG}}
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_name {{.*}}"foo"

; CHECK: Compile Unit:

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_declaration
; CHECK-NEXT: DW_AT_signature

; CHECK: DW_TAG_namespace
; CHECK-NOT: {{DW_AT_name|DW_TAG}}
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}}"foo"
; CHECK-NEXT: DW_AT_byte_size

%struct.bar = type { %"struct.(anonymous namespace)::foo" }
%"struct.(anonymous namespace)::foo" = type { i8 }

@b = global %struct.bar zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !13}
!llvm.ident = !{!12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "b", scope: !2, file: !3, line: 8, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 5.0.0 (trunk 294954) (llvm/trunk 294959)", isOptimized: false, runtimeVersion: 0, splitDebugFilename: "tu-to-non-tu.dwo", emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "tu.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "bar", file: !3, line: 5, size: 8, elements: !7, identifier: "_ZTS3bar")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !6, file: !3, line: 6, baseType: !9, size: 8)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", scope: !10, file: !3, line: 2, size: 8, elements: !4)
!10 = !DINamespace(scope: null)
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{!"clang version 5.0.0 (trunk 294954) (llvm/trunk 294959)"}
!13 = !{i32 2, !"Dwarf Version", i32 5}
