; XFAIL: target={{.*}}-aix{{.*}}
; RUN: %llc_dwarf -debugger-tune=lldb -accel-tables=Dwarf -filetype=obj -o %t %s
; RUN: llvm-dwarfdump -debug-names %t | FileCheck %s
; RUN: llvm-dwarfdump -debug-names -verify %t | FileCheck --check-prefix=VERIFY %s

@nameless_var = constant i8 0, !dbg !0

; CHECK: Name count: 0
; VERIFY: No errors

!llvm.dbg.cu = !{!10}
!llvm.module.flags = !{!14, !15}
!llvm.ident = !{!20}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(scope: null, file: !11, line: 1, type: !5, isLocal: true, isDefinition: true)
!5 = !DIBasicType(size: 8, encoding: DW_ATE_signed_char)
!10 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !11, producer: "blah", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !12, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/")
!11 = !DIFile(filename: "blah", directory: "blah")
!12 = !{!0}
!14 = !{i32 7, !"Dwarf Version", i32 5}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!20 = !{!"blah"}
