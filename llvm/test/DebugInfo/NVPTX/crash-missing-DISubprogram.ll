; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda
; Don't crash for a function w/o debug info that contains an instruction w/
; debug info.
; Reported as #51079

define weak void @test() {
  ret void, !dbg !10
}

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7}
!llvm.dbg.cu = !{!8}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 4]}
!1 = !{i32 7, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"openmp", i32 50}
!5 = !{i32 7, !"openmp-device", i32 50}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !9, producer: "clang version 14.0.0 (https://github.com/llvm/llvm-project.git a0262043bb87fdef68c817722de320a5dd9eb9c9)", isOptimized: true, runtimeVersion: 0, emissionKind: DebugDirectivesOnly, splitDebugInlining: false, nameTableKind: None)
!9 = !DIFile(filename: "test_omp_complex_reduction.cpp", directory: "/gpfs/jlse-fs0/users/jdoerfert/build/miniqmc-trunk/src/Platforms/tests/OMPTarget")
!10 = !DILocation(line: 69, column: 18, scope: !11)
!11 = distinct !DISubprogram(name: "__omp_outlined__1", scope: !12, file: !12, line: 68, type: !13, scopeLine: 68, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !8, retainedNodes: !14)
!12 = !DIFile(filename: "projects/benchmarks/miniqmc/src/Platforms/tests/OMPTarget/test_omp_complex_reduction.cpp", directory: "/gpfs/jlse-fs0/users/jdoerfert")
!13 = !DISubroutineType(types: !14)
!14 = !{}
