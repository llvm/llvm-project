; RUN: rm -rf %t && mkdir -p %t
; RUN: echo 'target datalayout = "e"' > %t/little.txt
; RUN: echo 'target datalayout = "E"' > %t/big.txt
; RUN: echo '!9 = !{!"%/t/version.ll", !0}' > %t/version.txt
; RUN: cat %t/little.txt %s %t/version.txt > %t/2

; RUN: opt -passes=insert-gcov-profiling -disable-output < %t/2
; RUN: head -c8 %t/version.gcno | grep '^oncg.11B'
; RUN: rm %t/version.gcno
; RUN: not opt -passes=insert-gcov-profiling -default-gcov-version=asdfasdf -disable-output < %t/2
; RUN: opt -passes=insert-gcov-profiling -default-gcov-version='B21*' -disable-output < %t/2
; RUN: head -c8 %t/version.gcno | grep '^oncg.12B'
; RUN: rm %t/version.gcno

; RUN: cat %t/big.txt %s %t/version.txt > %t/big.ll
; RUN: opt -passes=insert-gcov-profiling -disable-output < %t/big.ll
; RUN: head -c8 %t/version.gcno | grep '^gcnoB11.'

define void @test() !dbg !5 {
  ret void, !dbg !8
}

!llvm.gcov = !{!9}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 176994)", isOptimized: false, emissionKind: FullDebug, file: !11, enums: !3, retainedTypes: !3, globals: !3)
!2 = !DIFile(filename: "version", directory: "/usr/local/google/home/nlewycky")
!3 = !{}
!5 = distinct !DISubprogram(name: "test", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !10, scope: !6, type: !7, retainedNodes: !3)
!6 = !DIFile(filename: "<stdin>", directory: ".")
!7 = !DISubroutineType(types: !{null})
!8 = !DILocation(line: 1, scope: !5)
;; !9 is added through the echo line at the top.
!10 = !DIFile(filename: "<stdin>", directory: ".")
!11 = !DIFile(filename: "version", directory: "/usr/local/google/home/nlewycky")
!12 = !{i32 1, !"Debug Info Version", i32 3}
