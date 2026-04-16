; RUN: %llc_dwarf -dwarf-version=3 -O0 -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck --check-prefix=CHECK-V3 %s
; RUN: %llc_dwarf -dwarf-version=2 -O0 -filetype=obj < %s | llvm-dwarfdump -v -debug-info - | FileCheck --check-prefix=CHECK-V2 --implicit-check-not=DW_FORM_ref_addr %s

; DWARF V3: Expect cross-CU type deduplication resulting in only one "int"
; definition and a DW_FORM_ref_addr.

; CHECK-V3: DW_TAG_compile_unit
; CHECK-V3: DW_AT_name{{.*}}"a.cpp"
; CHECK-V3: DW_TAG_base_type
; CHECK-V3: DW_AT_name{{.*}}"int"
; CHECK-V3: DW_TAG_compile_unit
; CHECK-V3: DW_AT_name{{.*}}"b.cpp"
; CHECK-V3: DW_AT_type [DW_FORM_ref_addr]

; DWARF V2: cross-CU type deduplication disabled resulting in two "int"
; definitions and no DW_FORM_ref_addr (cross-CU ref)

; CHECK-V2: DW_TAG_compile_unit
; CHECK-V2: DW_AT_name{{.*}}"a.cpp"
; CHECK-V2: DW_TAG_base_type
; CHECK-V2: DW_AT_name{{.*}}"int"
; CHECK-V2: DW_TAG_compile_unit
; CHECK-V2: DW_AT_name{{.*}}"b.cpp"
; CHECK-V2: DW_TAG_base_type
; CHECK-V2: DW_AT_name{{.*}}"int"

@x = global i32 0, !dbg !0
@y = global i32 0, !dbg !7

!llvm.dbg.cu = !{!2, !9}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "a.cpp", directory: "/tmp/dbginfo")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = !DIGlobalVariableExpression(var: !8, expr: !DIExpression())
!8 = distinct !DIGlobalVariable(name: "y", scope: !9, file: !10, line: 1, type: !5, isLocal: false, isDefinition: true)
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !10, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !11)
!10 = !DIFile(filename: "b.cpp", directory: "/tmp/dbginfo")
!11 = !{!7}
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"clang"}
