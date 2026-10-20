; RUN: llc -mtriple=powerpc64-ibm-aix-xcoff -verify-machineinstrs %s -o - | FileCheck %s
; RUN: llc -mtriple=powerpc-ibm-aix-xcoff -verify-machineinstrs %s -o - | FileCheck %s

; CHECK:     .foo:
; CHECK:         bctr
; CHECK-NEXT: L..sec_end0:

@foo = ifunc i32 (...), ptr @foo.resolver

define internal ptr @foo.resolver() #0 !dbg !7 {
entry:
  ret ptr @my_foo2, !dbg !10
}

; Function Attrs: nounwind
define internal i32 @my_foo2() #1 !dbg !11 {
entry:
  ret i32 5, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 22.0.0git (git@github.ibm.com:compiler/llvm-project.git e963806df98c6d2e52573efbb8890ec72e5dd745)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "/home/wyehia/Source/scripts.fmv/fmv/ifunc/proto2")
!2 = !{i32 7, !"Dwarf Version", i32 3}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "foo_resolver", scope: !1, file: !1, line: 7, type: !8, scopeLine: 7, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!8 = !DISubroutineType(types: !9)
!9 = !{}
!10 = !DILocation(line: 7, scope: !7, atomGroup: 1, atomRank: 1)
!11 = distinct !DISubprogram(name: "my_foo2", scope: !1, file: !1, line: 2, type: !8, scopeLine: 2, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
!12 = !DILocation(line: 2, scope: !11, atomGroup: 1, atomRank: 1)
