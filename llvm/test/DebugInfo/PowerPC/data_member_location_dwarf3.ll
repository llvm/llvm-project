; RUN: llc -mtriple=powerpc64le-unknown-linux-gnu -dwarf-version=3 -O0 -o - -filetype=obj < %s | \
; RUN:   llvm-dwarfdump -v -debug-info -name g -| FileCheck %s

; CHECK: DW_AT_name [DW_FORM_strp] {{.*}} "g"
; CHECK: DW_AT_data_member_location [DW_FORM_udata] (65536)

; ModuleID = '1.cpp'
source_filename = "1.cpp"

%struct.e = type { [65536 x i8], i8 }

@E = dso_local local_unnamed_addr global %struct.e zeroinitializer, align 1, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "E", scope: !2, file: !3, line: 5, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "1.cpp", directory: "/tmp")
!4 = !{}
!5 = !{!0}
!6 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "e", file: !3, line: 1, size: 524296, flags: DIFlagTypePassByValue, elements: !7, identifier: "_ZTS1e")
!7 = !{!8, !13}
!8 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !6, file: !3, line: 2, baseType: !9, size: 524288)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 524288, elements: !11)
!10 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!11 = !{!12}
!12 = !DISubrange(count: 65536)
!13 = !DIDerivedType(tag: DW_TAG_member, name: "g", scope: !6, file: !3, line: 3, baseType: !10, size: 8, offset: 524288)
!14 = !{i32 7, !"Dwarf Version", i32 3}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{i32 7, !"uwtable", i32 1}
!18 = !{!"clang version 13.0.0"}
