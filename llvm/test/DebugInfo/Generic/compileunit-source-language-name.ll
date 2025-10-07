; RUN: %llc_dwarf -filetype=obj -O0 < %s | llvm-dwarfdump -debug-info - | FileCheck %s --implicit-check-not "DW_AT_language"

; CHECK: DW_AT_language_name (DW_LNAME_ObjC_plus_plus)

source_filename = "cu.cpp"
target triple = "arm64-apple-macosx"

@x = global i32 0, align 4, !dbg !0

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_ObjC_plus_plus, file: !3, producer: "handwritten", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!3 = !DIFile(filename: "cu.cpp", directory: "/tmp")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
