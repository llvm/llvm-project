; RUN: llc -filetype=obj -mtriple=powerpc64le-unknown-linux-gnu < %s | \
; RUN:   llvm-dwarfdump -debug-info - | FileCheck %s
; RUN: llc -filetype=obj -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -strict-dwarf=true < %s | llvm-dwarfdump -debug-info - | \
; RUN:   FileCheck %s

; FIXME: when -strict-dwarf=true is specified, we should check "STRICT" to tell
; that with DWARF 4, we should not generate DWARF 5 attribute DW_AT_noreturn and
; DW_AT_alignment.

; CHECK: DW_AT_alignment
; CHECK: DW_AT_noreturn
; STRICT-NOT: DW_AT_noreturn
; STRICT-NOT: DW_AT_alignment

@_ZL3var = internal global i32 0, align 16, !dbg !0

; Function Attrs: noinline noreturn optnone uwtable mustprogress
define dso_local void @_Z1fv() #0 !dbg !12 {
entry:
  call void @_Z4exitv(), !dbg !15
  ret void, !dbg !16
}

declare void @_Z4exitv()

; Function Attrs: noinline nounwind optnone uwtable mustprogress
define dso_local signext i32 @_Z3foov() !dbg !17 {
entry:
  %0 = load i32, i32* @_ZL3var, align 16, !dbg !21
  ret i32 %0, !dbg !22
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!7, !8, !9, !10}
!llvm.ident = !{!11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "var", linkageName: "_ZL3var", scope: !2, file: !3, line: 2, type: !6, isLocal: true, isDefinition: true, align: 128)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 13.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "t.cpp", directory: "./")
!4 = !{}
!5 = !{!0}
!6 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!7 = !{i32 7, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{i32 1, !"wchar_size", i32 4}
!10 = !{i32 7, !"uwtable", i32 1}
!11 = !{!"clang version 13.0.0"}
!12 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !3, file: !3, line: 4, type: !13, scopeLine: 4, flags: DIFlagPrototyped | DIFlagNoReturn, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!13 = !DISubroutineType(types: !14)
!14 = !{null}
!15 = !DILocation(line: 5, column: 1, scope: !12)
!16 = !DILocation(line: 7, column: 1, scope: !12)
!17 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !3, file: !3, line: 10, type: !18, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!18 = !DISubroutineType(types: !19)
!19 = !{!20}
!20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!21 = !DILocation(line: 12, column: 18, scope: !17)
!22 = !DILocation(line: 12, column: 11, scope: !17)
