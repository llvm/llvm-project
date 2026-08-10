; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=obj < %s \
; RUN: 	    | llvm-dwarfdump -v -debug-info - | FileCheck %s

; CHECK: DW_AT_location [DW_FORM_exprloc] (DW_OP_addrx 0x0)

%struct.foo = type { i32 }

@f = dso_local global %struct.foo zeroinitializer, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "f", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 8.0.0 (trunk 344833) (llvm/trunk 344837)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: GNU)
!3 = !DIFile(filename: "loc.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch", checksumkind: CSK_MD5, checksum: "e579a1a06fae14a4526216e905198a01")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "foo", file: !3, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS3foo")
!7 = !{!8}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "i", scope: !6, file: !3, line: 2, baseType: !9, size: 32)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{i32 2, !"Dwarf Version", i32 5}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 8.0.0 (trunk 344833) (llvm/trunk 344837)"}
