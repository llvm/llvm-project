; RUN: opt %loadPolly -polly-detect -polly-report -disable-output < %s  2>&1 | FileCheck %s
target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @foo(ptr %A) #0 !dbg !4 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body, !dbg !11

for.body:                                         ; preds = %entry.split, %for.body
  %indvar = phi i64 [ 0, %entry.split ], [ %indvar.next, %for.body ]
  %i.01 = trunc i64 %indvar to i32, !dbg !13
  %arrayidx = getelementptr float, ptr %A, i64 %indvar, !dbg !13
  %conv = sitofp i32 %i.01 to float, !dbg !13
  store float %conv, ptr %arrayidx, align 4, !dbg !13
  %indvar.next = add i64 %indvar, 1, !dbg !11
  %exitcond = icmp ne i64 %indvar.next, 100, !dbg !11
  br i1 %exitcond, label %for.body, label %for.end, !dbg !11

for.end:                                          ; preds = %for.body
  ret void, !dbg !14
}

; CHECK: note: Polly detected an optimizable loop region (scop) in function 'foo'
; CHECK: test.c:2: Start of scop
; CHECK: test.c:3: End of scop

; Function Attrs: nounwind uwtable
define void @bar(ptr %A) #0 !dbg !7 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br label %for.body, !dbg !15

for.body:                                         ; preds = %entry.split, %for.body
  %indvar = phi i64 [ 0, %entry.split ], [ %indvar.next, %for.body ]
  %i.01 = trunc i64 %indvar to i32, !dbg !17
  %arrayidx = getelementptr float, ptr %A, i64 %indvar, !dbg !17
  %conv = sitofp i32 %i.01 to float, !dbg !17
  store float %conv, ptr %arrayidx, align 4, !dbg !17
  %indvar.next = add i64 %indvar, 1, !dbg !15
  %exitcond = icmp ne i64 %indvar.next, 100, !dbg !15
  br i1 %exitcond, label %for.body, label %for.end, !dbg !15

for.end:                                          ; preds = %for.body
  ret void, !dbg !18
}

; CHECK: note: Polly detected an optimizable loop region (scop) in function 'bar'
; CHECK: test.c:9: Start of scop
; CHECK: test.c:13: End of scop

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 ", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "test.c", directory: "/home/grosser/Projects/polly/git/tools/polly")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "test.c", directory: "/home/grosser/Projects/polly/git/tools/polly")
!6 = !DISubroutineType(types: !{null})
!7 = distinct !DISubprogram(name: "bar", line: 6, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 6, file: !1, scope: !5, type: !6, retainedNodes: !2)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.5 "}
!11 = !DILocation(line: 2, scope: !12)
!12 = distinct !DILexicalBlock(line: 2, column: 0, file: !1, scope: !4)
!13 = !DILocation(line: 3, scope: !12)
!14 = !DILocation(line: 4, scope: !4)
!15 = !DILocation(line: 9, scope: !16)
!16 = distinct !DILexicalBlock(line: 9, column: 0, file: !1, scope: !7)
!17 = !DILocation(line: 13, scope: !16)
!18 = !DILocation(line: 14, scope: !7)

