; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/discriminator.prof | opt -analyze -branch-prob | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/discriminator.prof | opt -analyze -branch-prob | FileCheck %s

; Original code
;
; 1   int foo(int i) {
; 2     int x = 0;
; 3     while (i < 100) {
; 4       if (i < 5) x--;
; 5       i++;
; 6     }
; 7     return x;
; 8   }
;
; In this test, if the loop is executed 100 times, the decrement operation
; at line 4 should only execute 5 times. This is reflected in the profile
; data for line offset 3.  In Inputs/discriminator.prof, we have:
;
; 3: 100
; 3.1: 5
;
; This means that the predicate 'i < 5' (line 3) is executed 100 times,
; but the then branch (line 3.1) is only executed 5 times.

define i32 @foo(i32 %i) #0 !dbg !4 {
; CHECK: Printing analysis 'Branch Probability Analysis' for function 'foo':
entry:
  %i.addr = alloca i32, align 4
  %x = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  store i32 0, i32* %x, align 4, !dbg !10
  br label %while.cond, !dbg !11

while.cond:                                       ; preds = %if.end, %entry
  %0 = load i32, i32* %i.addr, align 4, !dbg !12
  %cmp = icmp slt i32 %0, 100, !dbg !12
  br i1 %cmp, label %while.body, label %while.end, !dbg !12
; CHECK: edge while.cond -> while.body probability is 0x7d83ba68 / 0x80000000 = 98.06% [HOT edge]
; CHECK: edge while.cond -> while.end probability is 0x027c4598 / 0x80000000 = 1.94%

while.body:                                       ; preds = %while.cond
  %1 = load i32, i32* %i.addr, align 4, !dbg !14
  %cmp1 = icmp slt i32 %1, 50, !dbg !14
  br i1 %cmp1, label %if.then, label %if.end, !dbg !14
; CHECK: edge while.body -> if.then probability is 0x07878788 / 0x80000000 = 5.88%
; CHECK: edge while.body -> if.end probability is 0x78787878 / 0x80000000 = 94.12% [HOT edge]

if.then:                                          ; preds = %while.body
  %2 = load i32, i32* %x, align 4, !dbg !17
  %dec = add nsw i32 %2, -1, !dbg !17
  store i32 %dec, i32* %x, align 4, !dbg !17
  br label %if.end, !dbg !17

if.end:                                           ; preds = %if.then, %while.body
  %3 = load i32, i32* %i.addr, align 4, !dbg !19
  %inc = add nsw i32 %3, 1, !dbg !19
  store i32 %inc, i32* %i.addr, align 4, !dbg !19
  br label %while.cond, !dbg !20

while.end:                                        ; preds = %while.cond
  %4 = load i32, i32* %x, align 4, !dbg !21
  ret i32 %4, !dbg !21
}


!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5 ", isOptimized: false, emissionKind: NoDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "discriminator.c", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !2)
!5 = !DIFile(filename: "discriminator.c", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 1, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.5 "}
!10 = !DILocation(line: 2, scope: !4)
!11 = !DILocation(line: 3, scope: !4)
!12 = !DILocation(line: 3, scope: !13)
!13 = !DILexicalBlockFile(discriminator: 2, file: !1, scope: !4)
!14 = !DILocation(line: 4, scope: !15)
!15 = distinct !DILexicalBlock(line: 4, column: 0, file: !1, scope: !16)
!16 = distinct !DILexicalBlock(line: 3, column: 0, file: !1, scope: !4)
!17 = !DILocation(line: 4, scope: !18)
!18 = !DILexicalBlockFile(discriminator: 2, file: !1, scope: !15)
!19 = !DILocation(line: 5, scope: !16)
!20 = !DILocation(line: 6, scope: !16)
!21 = !DILocation(line: 7, scope: !4)
