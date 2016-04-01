; RUN: llvm-as < %s | llvm-dis | FileCheck %s

!llvm.module.flags = !{!6}
!llvm.dbg.cu = !{!5}

!0 = !DISubprogram(name: "bar", linkageName: "_ZN3foo3barEv", line: 3, isLocal: false, isDefinition: false, virtualIndex: 6, flags: DIFlagProtected | DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !4, scope: !1, type: !2)
!1 = !DIFile(filename: "/foo", directory: "bar.cpp")
!2 = !DISubroutineType(types: !3)
!3 = !{null}
!4 = !DIFile(filename: "/foo", directory: "bar.cpp")
!5 = distinct !DICompileUnit(language: DW_LANG_C99, isOptimized: true, emissionKind: FullDebug, file: !4, subprograms: !{!0}, enums: !{}, retainedTypes: !{})

define <{i32, i32}> @f1() {
; CHECK: !dbgx ![[NUMBER:[0-9]+]]
  %r = insertvalue <{ i32, i32 }> zeroinitializer, i32 4, 1, !dbgx !1
; CHECK: !dbgx ![[NUMBER]]
  %e = extractvalue <{ i32, i32 }> %r, 0, !dbgx !1
  ret <{ i32, i32 }> %r
}

; CHECK: DIFlagProtected
!6 = !{i32 1, !"Debug Info Version", i32 3}
