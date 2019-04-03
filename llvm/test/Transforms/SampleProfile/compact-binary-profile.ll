; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/inline.prof -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline.prof -S | FileCheck %s
; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/inline.compactbinary.afdo -S | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/inline.compactbinary.afdo -S | FileCheck %s

; Original C++ test case
;
; #include <stdio.h>
;
; int sum(int x, int y) {
;   return x + y;
; }
;
; int main() {
;   int s, i = 0;
;   while (i++ < 20000 * 20000)
;     if (i != 100) s = sum(i, s); else s = 30;
;   printf("sum is %d\n", s);
;   return 0;
; }
;
@.str = private unnamed_addr constant [11 x i8] c"sum is %d\0A\00", align 1

; Check sample-profile phase using compactbinary format profile will annotate
; the IR with exactly the same result as using text format.
; CHECK: br i1 %cmp, label %while.body, label %while.end{{.*}} !prof ![[IDX1:[0-9]*]]
; CHECK: br i1 %cmp1, label %if.then, label %if.else{{.*}} !prof ![[IDX2:[0-9]*]]
; CHECK: call i32 (i8*, ...) @printf{{.*}} !prof ![[IDX3:[0-9]*]]
; CHECK: = !{!"TotalCount", i64 26781}
; CHECK: = !{!"MaxCount", i64 5553}
; CHECK: ![[IDX1]] = !{!"branch_weights", i32 5392, i32 163}
; CHECK: ![[IDX2]] = !{!"branch_weights", i32 5280, i32 113}
; CHECK: ![[IDX3]] = !{!"branch_weights", i32 1}

; Function Attrs: nounwind uwtable
define i32 @_Z3sumii(i32 %x, i32 %y) !dbg !4 {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %0 = load i32, i32* %x.addr, align 4, !dbg !11
  %1 = load i32, i32* %y.addr, align 4, !dbg !11
  %add = add nsw i32 %0, %1, !dbg !11
  ret i32 %add, !dbg !11
}

; Function Attrs: uwtable
define i32 @main() !dbg !7 {
entry:
  %retval = alloca i32, align 4
  %s = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, i32* %retval
  store i32 0, i32* %i, align 4, !dbg !12
  br label %while.cond, !dbg !13

while.cond:                                       ; preds = %if.end, %entry
  %0 = load i32, i32* %i, align 4, !dbg !14
  %inc = add nsw i32 %0, 1, !dbg !14
  store i32 %inc, i32* %i, align 4, !dbg !14
  %cmp = icmp slt i32 %0, 400000000, !dbg !14
  br i1 %cmp, label %while.body, label %while.end, !dbg !14

while.body:                                       ; preds = %while.cond
  %1 = load i32, i32* %i, align 4, !dbg !16
  %cmp1 = icmp ne i32 %1, 100, !dbg !16
  br i1 %cmp1, label %if.then, label %if.else, !dbg !16


if.then:                                          ; preds = %while.body
  %2 = load i32, i32* %i, align 4, !dbg !18
  %3 = load i32, i32* %s, align 4, !dbg !18
  %call = call i32 @_Z3sumii(i32 %2, i32 %3), !dbg !18
  store i32 %call, i32* %s, align 4, !dbg !18
  br label %if.end, !dbg !18

if.else:                                          ; preds = %while.body
  store i32 30, i32* %s, align 4, !dbg !20
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  br label %while.cond, !dbg !22

while.end:                                        ; preds = %while.cond
  %4 = load i32, i32* %s, align 4, !dbg !24
  %call2 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i32 %4), !dbg !24
  ret i32 0, !dbg !25
}

declare i32 @printf(i8*, ...) #2

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: NoDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "calls.cc", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "sum", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 3, file: !1, scope: !5, type: !6, retainedNodes: !2)
!5 = !DIFile(filename: "calls.cc", directory: ".")
!6 = !DISubroutineType(types: !2)
!7 = distinct !DISubprogram(name: "main", line: 7, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, scopeLine: 7, file: !1, scope: !5, type: !6, retainedNodes: !2)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 1, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.5 "}
!11 = !DILocation(line: 4, scope: !4)
!12 = !DILocation(line: 8, scope: !7)
!13 = !DILocation(line: 9, scope: !7)
!14 = !DILocation(line: 9, scope: !15)
!15 = !DILexicalBlockFile(discriminator: 2, file: !1, scope: !7)
!16 = !DILocation(line: 10, scope: !17)
!17 = distinct !DILexicalBlock(line: 10, column: 0, file: !1, scope: !7)
!18 = !DILocation(line: 10, scope: !19)
!19 = !DILexicalBlockFile(discriminator: 2, file: !1, scope: !17)
!20 = !DILocation(line: 10, scope: !21)
!21 = !DILexicalBlockFile(discriminator: 4, file: !1, scope: !17)
!22 = !DILocation(line: 10, scope: !23)
!23 = !DILexicalBlockFile(discriminator: 6, file: !1, scope: !17)
!24 = !DILocation(line: 11, scope: !7)
!25 = !DILocation(line: 12, scope: !7)
