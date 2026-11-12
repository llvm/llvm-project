; This test verifies the backward compatibility of DIStringType.
;; Specifically, it makes sure that bitcode for DIStringType without
;; the StringLocationExp field can be disassembled.
; REQUIRES: x86_64-linux

; RUN: llvm-dis -o - %s.bc | FileCheck %s

; CHECK: !DIStringType(name: ".str.DEFERRED", stringLengthExpression: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 8))

; ModuleID = 'distringtype-backward.bc'
source_filename = "distringtype.f90"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"QNCA_a0$i8*$rank0$" = type { i8*, i64, i64, i64, i64, i64 }

@"assumedlength_$DEFERRED" = internal global %"QNCA_a0$i8*$rank0$" zeroinitializer, !dbg !0
@0 = internal unnamed_addr constant i32 2

; Function Attrs: noinline nounwind optnone uwtable
define void @MAIN__() #0 !dbg !2 {
alloca_0:
  ret void
}

declare i32 @for_set_reentrancy(i32* nocapture readonly)

declare i32 @for_alloc_allocatable_handle(i64, i8** nocapture, i32, i8*)

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "intel-lang"="fortran" "target-cpu"="x86-64" }

!llvm.module.flags = !{!10, !11}
!llvm.dbg.cu = !{!6}
!omp_offload.info = !{}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "deferred", linkageName: "assumedlength_$DEFERRED", scope: !2, file: !3, line: 2, type: !9, isLocal: true, isDefinition: true)
!2 = distinct !DISubprogram(name: "assumedlength", linkageName: "MAIN__", scope: !3, file: !3, line: 1, type: !4, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !6, retainedNodes: !8)
!3 = !DIFile(filename: "test2.f90", directory: "/iusers/cchen15/examples/tests/jr33383")
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "Intel(R) Fortran 22.0-1258", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !7, splitDebugInlining: false, nameTableKind: None)
!7 = !{!0}
!8 = !{}
!9 = !DIStringType(name: ".str.DEFERRED", stringLengthExpression: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 8))
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = !{i32 2, !"Dwarf Version", i32 4}
