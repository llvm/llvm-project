; RUN: opt %s -passes=sample-profile -sample-profile-file=%S/Inputs/remap.prof -sample-profile-remapping-file=%S/Inputs/remap.map | opt -analyze -branch-prob | FileCheck %s

; Reduced from branch.ll

declare i1 @foo()

define void @_ZN3foo3barERKN1M1XINS_6detail3quxEEE() !dbg !2 {
; CHECK: Printing analysis 'Branch Probability Analysis' for function '_ZN3foo3barERKN1M1XINS_6detail3quxEEE':

entry:
  %cmp = call i1 @foo(), !dbg !6
  br i1 %cmp, label %if.then, label %if.end
; CHECK:  edge entry -> if.then probability is 0x4ccf6b16 / 0x80000000 = 60.01%
; CHECK:  edge entry -> if.end probability is 0x333094ea / 0x80000000 = 39.99%

if.then:
  br label %return

if.end:
  %cmp1 = call i1 @foo(), !dbg !7
  br i1 %cmp1, label %if.then.2, label %if.else
; CHECK: edge if.end -> if.then.2 probability is 0x6652c748 / 0x80000000 = 79.94%
; CHECK: edge if.end -> if.else probability is 0x19ad38b8 / 0x80000000 = 20.06%

if.then.2:
  call i1 @foo(), !dbg !8
  br label %for.cond

for.cond:
  %cmp5 = call i1 @foo()
  br i1 %cmp5, label %for.body, label %for.end, !prof !9
; CHECK: edge for.cond -> for.body probability is 0x73333333 / 0x80000000 = 90.00%
; CHECK: edge for.cond -> for.end probability is 0x0ccccccd / 0x80000000 = 10.00%

for.body:
  br label %for.cond

for.end:
  br label %return

if.else:
  br label %return

return:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "foo++", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !{}, retainedTypes: !{})
!1 = !DIFile(filename: "test.cc", directory: "/foo/bar")
!2 = distinct !DISubprogram(name: "_ZN3foo3barERKN1M1XINS_6detail3quxEEE", scope: !1, file: !1, line: 4, type: !3, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !{})
!3 = !DISubroutineType(types: !{})
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !DILocation(line: 5, column: 8, scope: !2)
!7 = !DILocation(line: 8, column: 6, scope: !2)
!8 = !DILocation(line: 10, column: 11, scope: !2)
!9 = !{!"branch_weights", i32 90, i32 10}
