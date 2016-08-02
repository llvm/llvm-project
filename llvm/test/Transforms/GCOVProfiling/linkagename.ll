; RUN: echo '!9 = !{!"%/T/linkagename.ll", !0}' > %t1
; RUN: cat %s %t1 > %t2
; RUN: opt -insert-gcov-profiling -disable-output < %t2
; RUN: grep _Z3foov %T/linkagename.gcno
; RUN: rm %T/linkagename.gcno

define void @_Z3foov() !dbg !5 {
entry:
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}
!llvm.gcov = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 177323)", isOptimized: false, emissionKind: 0, file: !2, enums: !3, retainedTypes: !3, subprograms: !4, globals: !3, imports:  !3)
!1 = !DIFile(filename: "hello.cc", directory: "/home/nlewycky")
!2 = !DIFile(filename: "hello.cc", directory: "/home/nlewycky")
!3 = !{}
!4 = !{!5}
!5 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !1, scope: !1, type: !6, variables: !3)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 1, scope: !5)


!10 = !{i32 1, !"Debug Info Version", i32 3}
