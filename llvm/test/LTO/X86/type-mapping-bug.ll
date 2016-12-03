; RUN: llvm-as -o %t.dst.bc %s
; RUN: llvm-as -o %t.src.bc %S/Inputs/type-mapping-src.ll
; RUN: llvm-lto %t.dst.bc %t.src.bc -o=/dev/null

target triple = "x86_64-pc-windows-msvc18.0.0"

; @x in Src will be linked with this @x, causing SrcType in Src to be mapped
; to %DstType.
%DstType = type { i8 }
@x = global %DstType zeroinitializer

; The Src module will re-use our DINode for this type.
%CommonStruct = type { i32 }
@foo = internal global %CommonStruct zeroinitializer, !dbg !5

; That DINode will refer to this value, casted to %Tricky.1* (!11),
; which will then show up in Src's getIdentifiedStructTypes().
@templateValueParam = global i8 zeroinitializer

; Because of the names, we would try to map %Tricky.1 to %Tricky --
; mapping a Dst type to another Dst type! This would assert when
; getting a mapping from %DstType, which has previously used as
; a destination type. Since these types are not in the source module,
; there should be no attempt to create a mapping involving them;
; both types should be left as they are.
%Tricky = type opaque
%Tricky.1 = type { %DstType* }


; Mark %Tricky used.
@use = global %Tricky* zeroinitializer

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!19}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "a", directory: "/")
!3 = !{}
!4 = !{!5}
!5 = distinct !DIGlobalVariable(name: "foo", linkageName: "foo", scope: !1, file: !2, line: 5, type: !6, isLocal: false, isDefinition: true)
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !2, line: 5, size: 8, elements: !7, identifier: ".?AUS@@")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !6, baseType: !9)
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Template<&x>", file: !2, line: 3, size: 8, elements: !3, templateParams: !10, identifier: ".?AU?$Template@$1?x@@3UX@@A@@")
!10 = !{!11}

!11 = !DITemplateValueParameter(type: !12, value: %Tricky.1* bitcast (i8* @templateValueParam to %Tricky.1*))

!12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "X", file: !2, line: 1, size: 8, elements: !3, identifier: ".?AUX@@")
!19 = !{i32 2, !"Debug Info Version", i32 3}
