; RUN: llc -mcpu=diamondrapids -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mcpu=diamondrapids -mtriple=x86_64-windows-msvc < %s -filetype=obj | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ
; RUN: llc -mcpu=novalake -mtriple=x86_64-windows-msvc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mcpu=novalake -mtriple=x86_64-windows-msvc < %s -filetype=obj | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; ASM: #DEBUG_VALUE: foo:bar <- $r16
; OBJ:      DefRangeRegisterSym {
; OBJ-NEXT:   Kind: S_DEFRANGE_REGISTER (0x1141)
; OBJ-NEXT:   Register: R16 (0x358)
; OBJ-NEXT:   MayHaveNoName: 0
; OBJ-NEXT:   LocalVariableAddrRange {
; OBJ-NEXT:     OffsetStart: .text+0x2A
; OBJ-NEXT:     ISectStart: 0x0
; OBJ-NEXT:     Range: 0x76
; OBJ-NEXT:   }
; OBJ-NEXT: }

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

define void @foo(ptr %D, ptr %X, ptr %Y, ptr %IB, ptr %DX, ptr %DA, ptr %SA, <4 x i1> %0, <4 x i1> %1) !dbg !4 {
alloca_0:
  %fetch = load i64, ptr null, align 1, !dbg !7
    #dbg_value(i64 %fetch, !8, !DIExpression(), !10)
  br label %next

next:                                             ; preds = %alloca_0
  %.pre = insertelement <4 x i64> zeroinitializer, i64 %fetch, i64 0
  %cmp.103 = icmp ugt <4 x i64> %.pre, zeroinitializer
  %2 = tail call <4 x double> @llvm.masked.load.v4f64.p0(ptr %SA, <4 x i1> %0, <4 x double> zeroinitializer)
  %3 = select <4 x i1> %cmp.103, <4 x double> %2, <4 x double> zeroinitializer
  %4 = tail call <4 x double> @llvm.masked.load.v4f64.p0(ptr %DX, <4 x i1> %0, <4 x double> zeroinitializer)
  %5 = tail call <4 x double> @llvm.masked.load.v4f64.p0(ptr %Y, <4 x i1> %0, <4 x double> zeroinitializer)
  %6 = fmul <4 x double> %4, %5
  %7 = tail call <4 x double> @llvm.masked.load.v4f64.p0(ptr %IB, <4 x i1> %0, <4 x double> zeroinitializer)
  %8 = fadd <4 x double> %6, %3
  %9 = fmul <4 x double> %7, %8
  %10 = tail call <4 x double> @llvm.masked.load.v4f64.p0(ptr %D, <4 x i1> %0, <4 x double> zeroinitializer)
  %11 = tail call <4 x double> @llvm.masked.load.v4f64.p0(ptr %DA, <4 x i1> %0, <4 x double> zeroinitializer)
  %12 = fmul <4 x double> %10, %11
  %13 = fadd <4 x double> %9, %12
  %14 = tail call double @llvm.vector.reduce.fadd.v4f64(double 0.000000e+00, <4 x double> %13)
  store double %14, ptr %X, align 1
  ret void
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.vector.reduce.fadd.v4f64(double, <4 x double>) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x double> @llvm.masked.load.v4f64.p0(ptr captures(none), <4 x i1>, <4 x double>) #1

; uselistorder directives
uselistorder ptr @llvm.masked.load.v4f64.p0, { 5, 4, 3, 2, 1, 0 }

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"CodeView", i32 1}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "foo", directory: "")
!4 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: !3, file: !3, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !6)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 2, scope: !4)
!8 = !DILocalVariable(name: "bar", scope: !4, type: !9, flags: DIFlagArtificial)
!9 = !DIBasicType(name: "INTEGER*8", size: 64, encoding: DW_ATE_signed)
!10 = !DILocation(line: 0, scope: !4)
