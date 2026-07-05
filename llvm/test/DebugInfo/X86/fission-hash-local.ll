; RUN: llc -split-dwarf-file=foo.dwo -O0 < %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - \
; RUN:   | llvm-dwarfdump - | FileCheck --check-prefix=H1 %s
; RUN: llc -split-dwarf-file=foo.dwo -O0 < %p/../Inputs/fission-hash-local2.ll -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o - \
; RUN:   | llvm-dwarfdump - | FileCheck --check-prefix=H2 %s

; Testing that the location of a local variable in a global function is hashed
; fission-hash-local2.ll is identical except for the value of the local
; variable (local.ll uses the constant 7 in the llvm.dbg.value below, local2.ll
; uses the constant 9) so it should have a different dwo_id, seen below.

; Original source:
; void f1() {
;   int i = 7; // or 9
; }

; H1: DW_AT_GNU_dwo_id  (0xb06b36aa6004befe)
; H2: DW_AT_GNU_dwo_id  (0xc3e980cf3dfa79a2)

; Function Attrs: norecurse nounwind readnone uwtable
define dso_local void @_Z2f1v() local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 7, metadata !11, metadata !DIExpression()), !dbg !13
  ret void, !dbg !14
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 11.0.0 (git@github.com:llvm/llvm-project.git edc3f4f02e54c2ae1067f60f6a0ed6caf5b92ef6)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "hash.cpp", directory: "/usr/local/google/home/blaikie/dev/scratch")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 11.0.0 (git@github.com:llvm/llvm-project.git edc3f4f02e54c2ae1067f60f6a0ed6caf5b92ef6)"}
!7 = distinct !DISubprogram(name: "f1", linkageName: "_Z2f1v", scope: !1, file: !1, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !{!11}
!11 = !DILocalVariable(name: "i", scope: !7, file: !1, line: 2, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DILocation(line: 0, scope: !7)
!14 = !DILocation(line: 3, column: 1, scope: !7)
