; This test verifies that the targetFuncName attribute in a DISubprogram
; is assembled/disassembled correctly.
;
; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
;
; CHECK: !DISubprogram(name: "sub1_.t0p", linkageName: "sub1_.t0p",{{.*}}, targetFuncName: "sub1_")
;
; ModuleID = 'main.f'
source_filename = "main.f"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal void @sub1_.t0p(ptr %arg0) #0 !dbg !23 {
wrap_start11:
  call void (...) @sub1_(ptr %arg0), !dbg !25
  ret void, !dbg !25
}

declare void @sub1_(...)

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "intel-lang"="fortran" "loopopt-pipeline"="light" "min-legal-vector-width"="0" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }

!llvm.module.flags = !{!12, !13}
!llvm.dbg.cu = !{!7}
!omp_offload.info = !{}

!4 = !DIFile(filename: "main.f", directory: "/dir")
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !4, producer: "Intel(R) Fortran 22.0-1483", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 2, !"Dwarf Version", i32 4}
!23 = distinct !DISubprogram(name: "sub1_.t0p", linkageName: "sub1_.t0p", scope: !4, file: !4, type: !5, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !7, retainedNodes: !24, targetFuncName: "sub1_")
!24 = !{}
!25 = !DILocation(line: 0, scope: !23)
