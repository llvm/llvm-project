; RUN: llc < %s -mcpu=sm_60 | FileCheck %s
; RUN: %if ptxas-sm_60 %{ llc < %s -mcpu=sm_60 | %ptxas-verify -arch=sm_60 %}
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Verify that the symbols with the characters illegal in PTX get appropriately mangled.
@__PRETTY_FUNCTION__._Z3foov = private unnamed_addr constant [11 x i8] c"void foo()\00", align 1, !dbg !0
; '.' gets replaced with `_$_`.
; CHECK: .global .align 1 .b8 __PRETTY_FUNCTION___$__Z3foov[11] = {118, 111, 105, 100, 32, 102, 111, 111, 40, 41};

; .debug* section names are special and are allowed to have the leading dot.
; CHECK-DAG: .section        .debug_abbrev
; CHECK-DAG: .section        .debug_info
; CHECK-DAG: .b32 .debug_abbrev
; CHECK-DAG: .b32 .debug_line
; CHECK-DAG: .section        .debug_macinfo

; .. but the symbol name must be mangled the same way here as it was at the definition point.
; CHECK-DAG: .b64 __PRETTY_FUNCTION___$__Z3foov
;

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!10, !11, !12}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(scope: null, file: !2, line: 1, type: !3, isLocal: true, isDefinition: true)
!2 = !DIFile(filename: "<stdin>", directory: "/usr/local/google/home/tra/work/llvm/build/release+assert+zapcc/dbg-dot")
!3 = !DICompositeType(tag: DW_TAG_array_type, baseType: !4, size: 88, elements: !6)
!4 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !5)
!5 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!6 = !{!7}
!7 = !DISubrange(count: 11)
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !2, producer: "clang version 20.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !9, splitDebugInlining: false, nameTableKind: None)
!9 = !{!0}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 1, !"wchar_size", i32 4}
!12 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!13 = !{!"clang version 20.0.0git"}
!14 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !2, file: !2, line: 1, type: !15, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !8)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !DILocation(line: 1, column: 56, scope: !14)
