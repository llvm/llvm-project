; RUN: llc -filetype=obj %s -o - | llvm-dwarfdump -  | FileCheck %s

; ModuleID = 'debugtest.c'
source_filename = "debugtest.c"
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32"
; Function Attrs: noinline nounwind optnone
define hidden i32 @foo(i32 %arg) #0 !dbg !7 {
entry:
  %arg.addr = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 %arg, i32* %arg.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %arg.addr, metadata !11, metadata !DIExpression()), !dbg !12
  call void @llvm.dbg.declare(metadata i32* %a, metadata !13, metadata !DIExpression()), !dbg !14
  store i32 1, i32* %a, align 4, !dbg !14
  call void @llvm.dbg.declare(metadata i32* %b, metadata !15, metadata !DIExpression()), !dbg !17
  store i32 2, i32* %b, align 4, !dbg !17
  %0 = load i32, i32* %b, align 4, !dbg !18
  store i32 %0, i32* %arg.addr, align 4, !dbg !19
  %1 = load i32, i32* %arg.addr, align 4, !dbg !20
  %2 = load i32, i32* %a, align 4, !dbg !21
  %add = add nsw i32 %1, %2, !dbg !22
  ret i32 %add, !dbg !23
}
; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git 6b38826e3a5f402498f0ea721b8c90d727f36205)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "debugtest.c", directory: "/s/llvm-upstream")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git 6b38826e3a5f402498f0ea721b8c90d727f36205)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "arg", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!12 = !DILocation(line: 1, column: 14, scope: !7)
!13 = !DILocalVariable(name: "a", scope: !7, file: !1, line: 2, type: !10)
!14 = !DILocation(line: 2, column: 7, scope: !7)
!15 = !DILocalVariable(name: "b", scope: !16, file: !1, line: 4, type: !10)
!16 = distinct !DILexicalBlock(scope: !7, file: !1, line: 3, column: 3)
!17 = !DILocation(line: 4, column: 9, scope: !16)
!18 = !DILocation(line: 5, column: 11, scope: !16)
!19 = !DILocation(line: 5, column: 9, scope: !16)
!20 = !DILocation(line: 7, column: 10, scope: !7)
!21 = !DILocation(line: 7, column: 16, scope: !7)
!22 = !DILocation(line: 7, column: 14, scope: !7)
!23 = !DILocation(line: 7, column: 3, scope: !7)
!24 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 10, type: !8, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!25 = !DILocalVariable(name: "arg", arg: 1, scope: !24, file: !1, line: 10, type: !10)
!26 = !DILocation(line: 10, column: 14, scope: !24)
!27 = !DILocation(line: 11, column: 10, scope: !24)
!28 = !DILocation(line: 11, column: 3, scope: !24)
!29 = !DILocalVariable(name: "__vla_expr0", scope: !24, type: !30, flags: DIFlagArtificial)
!30 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!31 = !DILocation(line: 0, scope: !24)
!32 = !DILocalVariable(name: "aa", scope: !24, file: !1, line: 11, type: !33)
!33 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, elements: !34)
!34 = !{!35}
!35 = !DISubrange(count: !29)
!36 = !DILocation(line: 11, column: 7, scope: !24)
!37 = !DILocalVariable(name: "cc", scope: !24, file: !1, line: 13, type: !38)
!38 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!39 = !DILocation(line: 13, column: 8, scope: !24)
!40 = !DILocation(line: 15, column: 1, scope: !24)

; CHECK-LABEL: DW_TAG_compile_unit
; CHECK-LABEL:     DW_TAG_subprogram
; CHECK-NEXT:                DW_AT_low_pc	(0x00000002)
; CHECK-NEXT:                DW_AT_high_pc	(0x00000039)
; CHECK-NEXT:                DW_AT_frame_base	(DW_OP_WASM_location 0x0 0x1, DW_OP_stack_value)
; CHECK-NEXT:                DW_AT_name	("foo")
; CHECK-NEXT:                DW_AT_decl_file	("/s/llvm-upstream{{(/|\\)}}debugtest.c")
; CHECK-NEXT:                DW_AT_decl_line	(1)
; CHECK-NEXT:                DW_AT_prototyped	(true)
; CHECK-NEXT:                DW_AT_type	(0x00000073 "int")
; CHECK-NEXT:                DW_AT_external	(true)
; CHECK-LABEL:     DW_TAG_formal_parameter
; CHECK-NEXT:                  DW_AT_location	(DW_OP_fbreg +12)
; CHECK-NEXT:                  DW_AT_name	("arg")
; CHECK-NEXT:                  DW_AT_decl_file	("/s/llvm-upstream{{(/|\\)}}debugtest.c")
; CHECK-NEXT:                  DW_AT_decl_line	(1)
; CHECK-NEXT:                  DW_AT_type	(0x00000073 "int")

; CHECK-LABEL:     DW_TAG_variable
; CHECK-NEXT:                  DW_AT_location	(DW_OP_fbreg +8)
; CHECK-NEXT:                  DW_AT_name	("a")
; CHECK-NEXT:                  DW_AT_decl_file	("/s/llvm-upstream{{(/|\\)}}debugtest.c")
; CHECK-NEXT:                  DW_AT_decl_line	(2)
; CHECK-NEXT:                  DW_AT_type	(0x00000073 "int")

; CHECK-LABEL:     DW_TAG_lexical_block
; CHECK-NEXT:                  DW_AT_low_pc	(0x0000001c)
; CHECK-NEXT:                  DW_AT_high_pc	(0x0000002d)

; CHECK-LABEL:       DW_TAG_variable
; CHECK-NEXT:                    DW_AT_location	(DW_OP_fbreg +4)
; CHECK-NEXT:                    DW_AT_name	("b")
; CHECK-NEXT:                    DW_AT_decl_file	("/s/llvm-upstream{{(/|\\)}}debugtest.c")
; CHECK-NEXT:                    DW_AT_decl_line	(4)
; CHECK-NEXT:                    DW_AT_type	(0x00000073 "int")
