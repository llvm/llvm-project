; RUN: llvm-link -S %s | FileCheck %s
; RUN: llvm-link -S %s --try-experimental-debuginfo-iterators | FileCheck %s

; Test that when a debug metadata use-before-def is run through llvm-link, the
; value reference is preserved. Tests both singular uses and DIArgList uses of
; the value.

; CHECK-LABEL: @test
; CHECK: call void @llvm.dbg.value(metadata i32 %A,
; CHECK-NEXT: call void @llvm.dbg.value(metadata !DIArgList(i32 0, i32 %A),
; CHECK-NEXT: %A =

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown-intelfpga"

define void @test() {
entry:
  call void @llvm.dbg.value(metadata i32 %A, metadata !5, metadata !DIExpression()), !dbg !10
  call void @llvm.dbg.value(metadata !DIArgList(i32 0, i32 %A), metadata !5, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_minus, DW_OP_stack_value)), !dbg !10
  %A = add i32 0, 1
  ret void
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "bogus", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "test")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !DILocalVariable(name: "A", arg: 2, scope: !6, file: !1, line: 60, type: !9)
!6 = distinct !DISubprogram(name: "test", linkageName: "_test", scope: !1, file: !1, line: 60, type: !7, scopeLine: 61, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !6)
