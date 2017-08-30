source_filename = "test/LTO/X86/Inputs/type-mapping-src.ll"
target triple = "x86_64-pc-windows-msvc18.0.0"

%SrcType = type { i8 }
%CommonStruct = type opaque

@x = external global %SrcType
@bar = internal global %CommonStruct* null, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "bar", linkageName: "bar", scope: !2, file: !3, line: 2, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "b", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64)
!7 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !3, line: 1, flags: DIFlagFwdDecl, identifier: ".?AUS@@")
!8 = !{i32 2, !"Debug Info Version", i32 3}
