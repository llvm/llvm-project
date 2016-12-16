target triple = "x86_64-pc-windows-msvc18.0.0"

%SrcType = type { i8 }
@x = external global %SrcType

%CommonStruct = type opaque
@bar = internal global %CommonStruct* null, !dbg !0


!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!12}
!0 = !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "bar", linkageName: "bar", scope: !1, file: !2, line: 2, type: !5, isLocal: false, isDefinition: true))
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, globals: !4)
!2 = !DIFile(filename: "b", directory: "/")
!3 = !{}
!4 = !{!0}
!5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !6, size: 64)
!6 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !2, line: 1, flags: DIFlagFwdDecl, identifier: ".?AUS@@")
!12 = !{i32 2, !"Debug Info Version", i32 3}

