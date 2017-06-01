; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/propagate.prof | opt -analyze -branch-prob | FileCheck %s
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%S/Inputs/propagate.prof | opt -analyze -branch-prob | FileCheck %s

; Original C++ code for this test case:
;
; #include <stdio.h>
;
; long foo(int x, int y, long N) {
;   if (x < y) {
;     return y - x;
;   } else {
;     for (long i = 0; i < N; i++) {
;       if (i > N / 3)
;         x--;
;       if (i > N / 4) {
;         y++;
;         x += 3;
;       } else {
;         for (unsigned j = 0; j < 100; j++) {
;           x += j;
;           y -= 3;
;         }
;       }
;     }
;   }
;   return y * x;
; }
;
; int main() {
;   int x = 5678;
;   int y = 1234;
;   long N = 9999999;
;   printf("foo(%d, %d, %ld) = %ld\n", x, y, N, foo(x, y, N));
;   return 0;
; }

; ModuleID = 'propagate.cc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [24 x i8] c"foo(%d, %d, %ld) = %ld\0A\00", align 1

; Function Attrs: nounwind uwtable
define i64 @_Z3fooiil(i32 %x, i32 %y, i64 %N) #0 !dbg !6 {
entry:
  %retval = alloca i64, align 8
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  %N.addr = alloca i64, align 8
  %i = alloca i64, align 8
  %j = alloca i64, align 8
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !11, metadata !12), !dbg !13
  store i32 %y, i32* %y.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %y.addr, metadata !14, metadata !12), !dbg !15
  store i64 %N, i64* %N.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %N.addr, metadata !16, metadata !12), !dbg !17
  %0 = load i32, i32* %x.addr, align 4, !dbg !18
  %1 = load i32, i32* %y.addr, align 4, !dbg !20
  %cmp = icmp slt i32 %0, %1, !dbg !21
  br i1 %cmp, label %if.then, label %if.else, !dbg !22

if.then:                                          ; preds = %entry
  %2 = load i32, i32* %y.addr, align 4, !dbg !23
  %3 = load i32, i32* %x.addr, align 4, !dbg !25
  %sub = sub nsw i32 %2, %3, !dbg !26
  %conv = sext i32 %sub to i64, !dbg !23
  store i64 %conv, i64* %retval, align 8, !dbg !27
  br label %return, !dbg !27

if.else:                                          ; preds = %entry
  call void @llvm.dbg.declare(metadata i64* %i, metadata !28, metadata !12), !dbg !31
  store i64 0, i64* %i, align 8, !dbg !31
  br label %for.cond, !dbg !32

for.cond:                                         ; preds = %for.inc17, %if.else
  %4 = load i64, i64* %i, align 8, !dbg !33
  %5 = load i64, i64* %N.addr, align 8, !dbg !36
  %cmp1 = icmp slt i64 %4, %5, !dbg !37
  br i1 %cmp1, label %for.body, label %for.end19, !dbg !38

for.body:                                         ; preds = %for.cond
  %6 = load i64, i64* %i, align 8, !dbg !39
  %7 = load i64, i64* %N.addr, align 8, !dbg !42
  %div = sdiv i64 %7, 3, !dbg !43
  %cmp2 = icmp sgt i64 %6, %div, !dbg !44
  br i1 %cmp2, label %if.then3, label %if.end, !dbg !45
; CHECK:  edge for.body -> if.then3 probability is 0x51292fa6 / 0x80000000 = 63.41%
; CHECK:  edge for.body -> if.end probability is 0x2ed6d05a / 0x80000000 = 36.59%

if.then3:                                         ; preds = %for.body
  %8 = load i32, i32* %x.addr, align 4, !dbg !46
  %dec = add nsw i32 %8, -1, !dbg !46
  store i32 %dec, i32* %x.addr, align 4, !dbg !46
  br label %if.end, !dbg !47

if.end:                                           ; preds = %if.then3, %for.body
  %9 = load i64, i64* %i, align 8, !dbg !48
  %10 = load i64, i64* %N.addr, align 8, !dbg !50
  %div4 = sdiv i64 %10, 4, !dbg !51
  %cmp5 = icmp sgt i64 %9, %div4, !dbg !52
  br i1 %cmp5, label %if.then6, label %if.else7, !dbg !53
; CHECK:  edge if.end -> if.then6 probability is 0x5d89d89e / 0x80000000 = 73.08%
; CHECK:  edge if.end -> if.else7 probability is 0x22762762 / 0x80000000 = 26.92%

if.then6:                                         ; preds = %if.end
  %11 = load i32, i32* %y.addr, align 4, !dbg !54
  %inc = add nsw i32 %11, 1, !dbg !54
  store i32 %inc, i32* %y.addr, align 4, !dbg !54
  %12 = load i32, i32* %x.addr, align 4, !dbg !56
  %add = add nsw i32 %12, 3, !dbg !56
  store i32 %add, i32* %x.addr, align 4, !dbg !56
  br label %if.end16, !dbg !57

if.else7:                                         ; preds = %if.end
  call void @llvm.dbg.declare(metadata i64* %j, metadata !58, metadata !12), !dbg !62
  store i64 0, i64* %j, align 8, !dbg !62
  br label %for.cond8, !dbg !63

for.cond8:                                        ; preds = %for.inc, %if.else7
  %13 = load i64, i64* %j, align 8, !dbg !64
  %cmp9 = icmp slt i64 %13, 100, !dbg !67
  br i1 %cmp9, label %for.body10, label %for.end, !dbg !68
; CHECK: edge for.cond8 -> for.body10 probability is 0x7e941a89 / 0x80000000 = 98.89% [HOT edge]
; CHECK: edge for.cond8 -> for.end probability is 0x016be577 / 0x80000000 = 1.11%


for.body10:                                       ; preds = %for.cond8
  %14 = load i64, i64* %j, align 8, !dbg !69
  %15 = load i32, i32* %x.addr, align 4, !dbg !71
  %conv11 = sext i32 %15 to i64, !dbg !71
  %add12 = add nsw i64 %conv11, %14, !dbg !71
  %conv13 = trunc i64 %add12 to i32, !dbg !71
  store i32 %conv13, i32* %x.addr, align 4, !dbg !71
  %16 = load i32, i32* %y.addr, align 4, !dbg !72
  %sub14 = sub nsw i32 %16, 3, !dbg !72
  store i32 %sub14, i32* %y.addr, align 4, !dbg !72
  br label %for.inc, !dbg !73

for.inc:                                          ; preds = %for.body10
  %17 = load i64, i64* %j, align 8, !dbg !74
  %inc15 = add nsw i64 %17, 1, !dbg !74
  store i64 %inc15, i64* %j, align 8, !dbg !74
  br label %for.cond8, !dbg !76

for.end:                                          ; preds = %for.cond8
  br label %if.end16

if.end16:                                         ; preds = %for.end, %if.then6
  br label %for.inc17, !dbg !77

for.inc17:                                        ; preds = %if.end16
  %18 = load i64, i64* %i, align 8, !dbg !78
  %inc18 = add nsw i64 %18, 1, !dbg !78
  store i64 %inc18, i64* %i, align 8, !dbg !78
  br label %for.cond, !dbg !80

for.end19:                                        ; preds = %for.cond
  br label %if.end20

if.end20:                                         ; preds = %for.end19
  %19 = load i32, i32* %y.addr, align 4, !dbg !81
  %20 = load i32, i32* %x.addr, align 4, !dbg !82
  %mul = mul nsw i32 %19, %20, !dbg !83
  %conv21 = sext i32 %mul to i64, !dbg !81
  store i64 %conv21, i64* %retval, align 8, !dbg !84
  br label %return, !dbg !84

return:                                           ; preds = %if.end20, %if.then
  %21 = load i64, i64* %retval, align 8, !dbg !85
  ret i64 %21, !dbg !85
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: norecurse uwtable
define i32 @main() #2 !dbg !86 {
entry:
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  %N = alloca i64, align 8
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata i32* %x, metadata !89, metadata !12), !dbg !90
  store i32 5678, i32* %x, align 4, !dbg !90
  call void @llvm.dbg.declare(metadata i32* %y, metadata !91, metadata !12), !dbg !92
  store i32 1234, i32* %y, align 4, !dbg !92
  call void @llvm.dbg.declare(metadata i64* %N, metadata !93, metadata !12), !dbg !94
  store i64 9999999, i64* %N, align 8, !dbg !94
  %0 = load i32, i32* %x, align 4, !dbg !95
  %1 = load i32, i32* %y, align 4, !dbg !96
  %2 = load i64, i64* %N, align 8, !dbg !97
  %3 = load i32, i32* %x, align 4, !dbg !98
  %4 = load i32, i32* %y, align 4, !dbg !99
  %5 = load i64, i64* %N, align 8, !dbg !100
  %call = call i64 @_Z3fooiil(i32 %3, i32 %4, i64 %5), !dbg !101
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @.str, i32 0, i32 0), i32 %0, i32 %1, i64 %2, i64 %call), !dbg !102
  ret i32 0, !dbg !104
}

declare i32 @printf(i8*, ...) #3

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { norecurse uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 266819)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "propagate.cc", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 266819)"}
!6 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooiil", scope: !1, file: !1, line: 3, type: !7, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !10, !10, !9}
!9 = !DIBasicType(name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "x", arg: 1, scope: !6, file: !1, line: 3, type: !10)
!12 = !DIExpression()
!13 = !DILocation(line: 3, column: 14, scope: !6)
!14 = !DILocalVariable(name: "y", arg: 2, scope: !6, file: !1, line: 3, type: !10)
!15 = !DILocation(line: 3, column: 21, scope: !6)
!16 = !DILocalVariable(name: "N", arg: 3, scope: !6, file: !1, line: 3, type: !9)
!17 = !DILocation(line: 3, column: 29, scope: !6)
!18 = !DILocation(line: 4, column: 7, scope: !19)
!19 = distinct !DILexicalBlock(scope: !6, file: !1, line: 4, column: 7)
!20 = !DILocation(line: 4, column: 11, scope: !19)
!21 = !DILocation(line: 4, column: 9, scope: !19)
!22 = !DILocation(line: 4, column: 7, scope: !6)
!23 = !DILocation(line: 5, column: 12, scope: !24)
!24 = distinct !DILexicalBlock(scope: !19, file: !1, line: 4, column: 14)
!25 = !DILocation(line: 5, column: 16, scope: !24)
!26 = !DILocation(line: 5, column: 14, scope: !24)
!27 = !DILocation(line: 5, column: 5, scope: !24)
!28 = !DILocalVariable(name: "i", scope: !29, file: !1, line: 7, type: !9)
!29 = distinct !DILexicalBlock(scope: !30, file: !1, line: 7, column: 5)
!30 = distinct !DILexicalBlock(scope: !19, file: !1, line: 6, column: 10)
!31 = !DILocation(line: 7, column: 15, scope: !29)
!32 = !DILocation(line: 7, column: 10, scope: !29)
!33 = !DILocation(line: 7, column: 22, scope: !34)
!34 = !DILexicalBlockFile(scope: !35, file: !1, discriminator: 1)
!35 = distinct !DILexicalBlock(scope: !29, file: !1, line: 7, column: 5)
!36 = !DILocation(line: 7, column: 26, scope: !34)
!37 = !DILocation(line: 7, column: 24, scope: !34)
!38 = !DILocation(line: 7, column: 5, scope: !34)
!39 = !DILocation(line: 8, column: 11, scope: !40)
!40 = distinct !DILexicalBlock(scope: !41, file: !1, line: 8, column: 11)
!41 = distinct !DILexicalBlock(scope: !35, file: !1, line: 7, column: 34)
!42 = !DILocation(line: 8, column: 15, scope: !40)
!43 = !DILocation(line: 8, column: 17, scope: !40)
!44 = !DILocation(line: 8, column: 13, scope: !40)
!45 = !DILocation(line: 8, column: 11, scope: !41)
!46 = !DILocation(line: 9, column: 10, scope: !40)
!47 = !DILocation(line: 9, column: 9, scope: !40)
!48 = !DILocation(line: 10, column: 11, scope: !49)
!49 = distinct !DILexicalBlock(scope: !41, file: !1, line: 10, column: 11)
!50 = !DILocation(line: 10, column: 15, scope: !49)
!51 = !DILocation(line: 10, column: 17, scope: !49)
!52 = !DILocation(line: 10, column: 13, scope: !49)
!53 = !DILocation(line: 10, column: 11, scope: !41)
!54 = !DILocation(line: 11, column: 10, scope: !55)
!55 = distinct !DILexicalBlock(scope: !49, file: !1, line: 10, column: 22)
!56 = !DILocation(line: 12, column: 11, scope: !55)
!57 = !DILocation(line: 13, column: 7, scope: !55)
!58 = !DILocalVariable(name: "j", scope: !59, file: !1, line: 14, type: !61)
!59 = distinct !DILexicalBlock(scope: !60, file: !1, line: 14, column: 9)
!60 = distinct !DILexicalBlock(scope: !49, file: !1, line: 13, column: 14)
!61 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!62 = !DILocation(line: 14, column: 24, scope: !59)
!63 = !DILocation(line: 14, column: 14, scope: !59)
!64 = !DILocation(line: 14, column: 31, scope: !65)
!65 = !DILexicalBlockFile(scope: !66, file: !1, discriminator: 1)
!66 = distinct !DILexicalBlock(scope: !59, file: !1, line: 14, column: 9)
!67 = !DILocation(line: 14, column: 33, scope: !65)
!68 = !DILocation(line: 14, column: 9, scope: !65)
!69 = !DILocation(line: 15, column: 16, scope: !70)
!70 = distinct !DILexicalBlock(scope: !66, file: !1, line: 14, column: 45)
!71 = !DILocation(line: 15, column: 13, scope: !70)
!72 = !DILocation(line: 16, column: 13, scope: !70)
!73 = !DILocation(line: 17, column: 9, scope: !70)
!74 = !DILocation(line: 14, column: 41, scope: !75)
!75 = !DILexicalBlockFile(scope: !66, file: !1, discriminator: 2)
!76 = !DILocation(line: 14, column: 9, scope: !75)
!77 = !DILocation(line: 19, column: 5, scope: !41)
!78 = !DILocation(line: 7, column: 30, scope: !79)
!79 = !DILexicalBlockFile(scope: !35, file: !1, discriminator: 2)
!80 = !DILocation(line: 7, column: 5, scope: !79)
!81 = !DILocation(line: 21, column: 10, scope: !6)
!82 = !DILocation(line: 21, column: 14, scope: !6)
!83 = !DILocation(line: 21, column: 12, scope: !6)
!84 = !DILocation(line: 21, column: 3, scope: !6)
!85 = !DILocation(line: 22, column: 1, scope: !6)
!86 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 24, type: !87, isLocal: false, isDefinition: true, scopeLine: 24, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!87 = !DISubroutineType(types: !88)
!88 = !{!10}
!89 = !DILocalVariable(name: "x", scope: !86, file: !1, line: 25, type: !10)
!90 = !DILocation(line: 25, column: 7, scope: !86)
!91 = !DILocalVariable(name: "y", scope: !86, file: !1, line: 26, type: !10)
!92 = !DILocation(line: 26, column: 7, scope: !86)
!93 = !DILocalVariable(name: "N", scope: !86, file: !1, line: 27, type: !9)
!94 = !DILocation(line: 27, column: 8, scope: !86)
!95 = !DILocation(line: 28, column: 38, scope: !86)
!96 = !DILocation(line: 28, column: 41, scope: !86)
!97 = !DILocation(line: 28, column: 44, scope: !86)
!98 = !DILocation(line: 28, column: 51, scope: !86)
!99 = !DILocation(line: 28, column: 54, scope: !86)
!100 = !DILocation(line: 28, column: 57, scope: !86)
!101 = !DILocation(line: 28, column: 47, scope: !86)
!102 = !DILocation(line: 28, column: 3, scope: !103)
!103 = !DILexicalBlockFile(scope: !86, file: !1, discriminator: 1)
!104 = !DILocation(line: 29, column: 3, scope: !86)
