; The two profiles used in this test are the same but encoded in different
; formats. This checks that we produce the same profile annotations regardless
; of the profile format.
;
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/fnptr.prof | opt -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/fnptr.binprof | opt -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

; CHECK:   edge for.body3 -> if.then probability is 0x1a56a56a / 0x80000000 = 20.58%
; CHECK:   edge for.body3 -> if.else probability is 0x65a95a96 / 0x80000000 = 79.42%
; CHECK:   edge for.inc -> for.inc12 probability is 0x000fbd1c / 0x80000000 = 0.05%
; CHECK:   edge for.inc -> for.body3 probability is 0x7ff042e4 / 0x80000000 = 99.95%
; CHECK:   edge for.inc12 -> for.end14 probability is 0x04000000 / 0x80000000 = 3.12%
; CHECK:   edge for.inc12 -> for.cond1.preheader probability is 0x7c000000 / 0x80000000 = 96.88%

; Original C++ test case.
;
; #include <stdlib.h>
; #include <math.h>
; #include <stdio.h>
;
; #define N 10000
; #define M 6000
;
; double foo(int x) {
;   return x * sin((double)x);
; }
;
; double bar(int x) {
;   return x - cos((double)x);
; }
;
; int main() {
;   double (*fptr)(int);
;   double S = 0;
;   for (int i = 0; i < N; i++)
;     for (int j = 0; j < M; j++) {
;       fptr = (rand() % 100 < 30) ? foo : bar;
;       if (rand() % 100 < 10)
;         S += (*fptr)(i + j * 300);
;       else
;         S += (*fptr)(i - j / 840);
;     }
;   printf("S = %lf\n", S);
;   return 0;
; }

@.str = private unnamed_addr constant [9 x i8] c"S = %lf\0A\00", align 1

define double @_Z3fooi(i32 %x) #0 !dbg !3 {
entry:
  %conv = sitofp i32 %x to double, !dbg !2
  %call = tail call double @sin(double %conv) #3, !dbg !8
  %mul = fmul double %conv, %call, !dbg !8
  ret double %mul, !dbg !8
}

declare double @sin(double) #1

define double @_Z3bari(i32 %x) #0 !dbg !10 {
entry:
  %conv = sitofp i32 %x to double, !dbg !9
  %call = tail call double @cos(double %conv) #3, !dbg !11
  %sub = fsub double %conv, %call, !dbg !11
  ret double %sub, !dbg !11
}

declare double @cos(double) #1

define i32 @main() #2 !dbg !13 {
entry:
  br label %for.cond1.preheader, !dbg !12

for.cond1.preheader:                              ; preds = %for.inc12, %entry
  %i.025 = phi i32 [ 0, %entry ], [ %inc13, %for.inc12 ]
  %S.024 = phi double [ 0.000000e+00, %entry ], [ %S.2.lcssa, %for.inc12 ]
  br label %for.body3, !dbg !14

for.body3:                                        ; preds = %for.inc, %for.cond1.preheader
  %j.023 = phi i32 [ 0, %for.cond1.preheader ], [ %inc, %for.inc ]
  %S.122 = phi double [ %S.024, %for.cond1.preheader ], [ %S.2, %for.inc ]
  %call = tail call i32 @rand() #3, !dbg !15
  %rem = srem i32 %call, 100, !dbg !15
  %cmp4 = icmp slt i32 %rem, 30, !dbg !15
  %_Z3fooi._Z3bari = select i1 %cmp4, double (i32)* @_Z3fooi, double (i32)* @_Z3bari, !dbg !15
  %call5 = tail call i32 @rand() #3, !dbg !16
  %rem6 = srem i32 %call5, 100, !dbg !16
  %cmp7 = icmp slt i32 %rem6, 10, !dbg !16
  br i1 %cmp7, label %if.then, label %if.else, !dbg !16

if.then:                                          ; preds = %for.body3
  %mul = mul nsw i32 %j.023, 300, !dbg !18
  %add = add nsw i32 %mul, %i.025, !dbg !18
  %call8 = tail call double %_Z3fooi._Z3bari(i32 %add), !dbg !18
  br label %for.inc, !dbg !18

if.else:                                          ; preds = %for.body3
  %div = sdiv i32 %j.023, 840, !dbg !19
  %sub = sub nsw i32 %i.025, %div, !dbg !19
  %call10 = tail call double %_Z3fooi._Z3bari(i32 %sub), !dbg !19
  br label %for.inc

for.inc:                                          ; preds = %if.then, %if.else
  %call8.pn = phi double [ %call8, %if.then ], [ %call10, %if.else ]
  %S.2 = fadd double %S.122, %call8.pn, !dbg !18
  %inc = add nsw i32 %j.023, 1, !dbg !20
  %exitcond = icmp eq i32 %j.023, 5999, !dbg !14
  br i1 %exitcond, label %for.inc12, label %for.body3, !dbg !14

for.inc12:                                        ; preds = %for.inc
  %S.2.lcssa = phi double [ %S.2, %for.inc ]
  %inc13 = add nsw i32 %i.025, 1, !dbg !22
  %exitcond26 = icmp eq i32 %i.025, 9999, !dbg !12
  br i1 %exitcond26, label %for.end14, label %for.cond1.preheader, !dbg !12

for.end14:                                        ; preds = %for.inc12
  %S.2.lcssa.lcssa = phi double [ %S.2.lcssa, %for.inc12 ]
  %call15 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i64 0, i64 0), double %S.2.lcssa.lcssa), !dbg !24
  ret i32 0, !dbg !25
}

; Function Attrs: nounwind
declare i32 @rand() #1

; Function Attrs: nounwind
declare i32 @printf(i8* nocapture readonly, ...) #1

attributes #0 = {"use-sample-profile"}
attributes #2 = {"use-sample-profile"}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}
!llvm.dbg.cu = !{!26}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{!"clang version 3.6.0 "}
!2 = !DILocation(line: 9, column: 3, scope: !3)
!3 = distinct !DISubprogram(name: "foo", line: 8, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !26, scopeLine: 8, file: !4, scope: !5, type: !6, retainedNodes: !7)
!4 = !DIFile(filename: "fnptr.cc", directory: ".")
!5 = !DIFile(filename: "fnptr.cc", directory: ".")
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 9, column: 14, scope: !3)
!9 = !DILocation(line: 13, column: 3, scope: !10)
!10 = distinct !DISubprogram(name: "bar", line: 12, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !26, scopeLine: 12, file: !4, scope: !5, type: !6, retainedNodes: !7)
!11 = !DILocation(line: 13, column: 14, scope: !10)
!12 = !DILocation(line: 19, column: 3, scope: !13)
!13 = distinct !DISubprogram(name: "main", line: 16, isLocal: false, isDefinition: true, flags: DIFlagPrototyped, isOptimized: true, unit: !26, scopeLine: 16, file: !4, scope: !5, type: !6, retainedNodes: !7)
!14 = !DILocation(line: 20, column: 5, scope: !13)
!15 = !DILocation(line: 21, column: 15, scope: !13)
!16 = !DILocation(line: 22, column: 11, scope: !13)
!18 = !DILocation(line: 23, column: 14, scope: !13)
!19 = !DILocation(line: 25, column: 14, scope: !13)
!20 = !DILocation(line: 20, column: 28, scope: !13)
!22 = !DILocation(line: 19, column: 26, scope: !13)
!24 = !DILocation(line: 27, column: 3, scope: !13)
!25 = !DILocation(line: 28, column: 3, scope: !13)
!26 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5 ", isOptimized: false, emissionKind: FullDebug, file: !4)
