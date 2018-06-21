; RUN: opt -newgvn -S < %s | FileCheck %s

; Check that the redundant load from %if.then is removed.
; Also, check that the debug location associated to load %0 still refers to
; line 3 and not line 6.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: @test_redundant_load(
; CHECK-LABEL: entry:
; CHECK-NEXT: load i32, i32* %Y, align 4, !dbg ![[LOC:[0-9]+]]
; CHECK-LABEL: if.then:
; CHECK-NOT: load
; CHECK-LABEL: if.end:
; CHECK: ![[LOC]] = !DILocation(line: 3, scope: !{{.*}})

define i32 @test_redundant_load(i32 %X, i32* %Y) !dbg !6 {
entry:
  %0 = load i32, i32* %Y, align 4, !dbg !8
  %cmp = icmp sgt i32 %X, -1, !dbg !9
  br i1 %cmp, label %if.then, label %if.end, !dbg !9

if.then:                                          ; preds = %entry
  %1 = load i32, i32* %Y, align 4, !dbg !10
  %add = add nsw i32 %0, %1, !dbg !10
  call void @foo(), !dbg !11
  br label %if.end, !dbg !12

if.end:                                           ; preds = %if.then, %entry
  %Result.0 = phi i32 [ %add, %if.then ], [ %0, %entry ]
  ret i32 %Result.0, !dbg !13
}

declare void @foo()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2)
!1 = !DIFile(filename: "test.cpp", directory: "")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = distinct !DISubprogram(name: "test_redundant_load", scope: !1, file: !1, line: 2, type: !7, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !2)
!8 = !DILocation(line: 3, scope: !6)
!9 = !DILocation(line: 5, scope: !6)
!10 = !DILocation(line: 6, scope: !6)
!11 = !DILocation(line: 7, scope: !6)
!12 = !DILocation(line: 8, scope: !6)
!13 = !DILocation(line: 10, scope: !6)
