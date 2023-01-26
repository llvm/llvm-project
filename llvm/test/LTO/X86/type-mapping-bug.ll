; RUN: llvm-as -o %t.dst.bc %s
; RUN: llvm-as -o %t.src.bc %S/Inputs/type-mapping-src.ll
; RUN: llvm-lto %t.dst.bc %t.src.bc -o=%t.lto.bc

source_filename = "test/LTO/X86/type-mapping-bug.ll"
target triple = "x86_64-pc-windows-msvc18.0.0"
; @x in Src will be linked with this @x, causing SrcType in Src to be mapped
; to %DstType.

%DstType = type { i8 }
%CommonStruct = type { i32 }
; The Src module will re-use our DINode for this type.
%Tricky = type opaque
%Tricky.1 = type { ptr }

@x = global %DstType zeroinitializer
@foo = internal global %CommonStruct zeroinitializer, !dbg !0
; That DINode will refer to this value, casted to ptr (!11),
; which will then show up in Src's getIdentifiedStructTypes().
@templateValueParam = global i8 0
; Because of the names, we would try to map %Tricky.1 to %Tricky --
; mapping a Dst type to another Dst type! This would assert when
; getting a mapping from %DstType, which has previously used as
; a destination type. Since these types are not in the source module,
; there should be no attempt to create a mapping involving them;
; both types should be left as they are.
@use = global ptr null

; Mark %Tricky used.
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "foo", linkageName: "foo", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "a", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 5, size: 8, elements: !7, identifier: ".?AUS@@")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !9)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Template<&x>", file: !3, line: 3, size: 8, elements: !4, templateParams: !10, identifier: ".?AU?$Template@$1?x@@3UX@@A@@")
!10 = !{!11}
!11 = !DITemplateValueParameter(type: !12, value: ptr @templateValueParam)
!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X", file: !3, line: 1, size: 8, elements: !4, identifier: ".?AUX@@")
!14 = !{i32 2, !"Debug Info Version", i32 3}

