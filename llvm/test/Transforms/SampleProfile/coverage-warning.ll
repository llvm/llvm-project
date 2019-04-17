; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/coverage-warning.prof -sample-profile-check-record-coverage=90 -sample-profile-check-sample-coverage=100 -o /dev/null 2>&1 | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/coverage-warning.prof -sample-profile-check-record-coverage=90 -sample-profile-check-sample-coverage=100 -o /dev/null 2>&1 | FileCheck %s
define i32 @foo(i32 %i) !dbg !4 {
; The profile has samples for line locations that are no longer present.
; Coverage does not reach 90%, so we should get this warning:
;
; CHECK: warning: coverage-warning.c:1: 2 of 3 available profile records (66%) were applied
; CHECK: warning: coverage-warning.c:1: 29000 of 30700 available profile samples (94%) were applied
entry:
  %retval = alloca i32, align 4
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  %0 = load i32, i32* %i.addr, align 4, !dbg !9
  %cmp = icmp sgt i32 %0, 1000, !dbg !10
  br i1 %cmp, label %if.then, label %if.end, !dbg !9

if.then:                                          ; preds = %entry
  store i32 30, i32* %retval, align 4, !dbg !11
  br label %return, !dbg !11

if.end:                                           ; preds = %entry
  store i32 3, i32* %retval, align 4, !dbg !12
  br label %return, !dbg !12

return:                                           ; preds = %if.end, %if.then
  %1 = load i32, i32* %retval, align 4, !dbg !13
  ret i32 %1, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 251524) (llvm/trunk 251531)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "coverage-warning.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !2)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang version 3.8.0 (trunk 251524) (llvm/trunk 251531)"}
!9 = !DILocation(line: 2, column: 7, scope: !4)
!10 = !DILocation(line: 2, column: 9, scope: !4)
!11 = !DILocation(line: 3, column: 5, scope: !4)
!12 = !DILocation(line: 4, column: 3, scope: !4)
!13 = !DILocation(line: 5, column: 1, scope: !4)
