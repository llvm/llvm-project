; RUN: opt < %s -instcombine -sample-profile -sample-profile-file=%S/Inputs/inline-coverage.prof -sample-profile-check-record-coverage=100 -sample-profile-check-sample-coverage=110 -pass-remarks=sample-profile -o /dev/null 2>&1 | FileCheck %s
; RUN: opt < %s -passes="function(instcombine),sample-profile" -sample-profile-file=%S/Inputs/inline-coverage.prof -sample-profile-check-record-coverage=100 -sample-profile-check-sample-coverage=110 -pass-remarks=sample-profile -o /dev/null 2>&1 | FileCheck %s
;
; Original code:
;
;     1    #include <stdlib.h>
;     2
;     3    long long int foo(long i) {
;     4      return rand() * i;
;     5    }
;     6
;     7    int main() {
;     8      long long int sum = 0;
;     9      for (int i = 0; i < 200000 * 3000; i++)
;    10        sum += foo(i);
;    11      return sum > 0 ? 0 : 1;
;    12    }
;
; CHECK: remark: coverage.cc:10:12: inlined hot callee '_Z3fool' with 172746 samples into 'main'
; CHECK: remark: coverage.cc:9:21: Applied 23478 samples from profile (offset: 2.1)
; CHECK: remark: coverage.cc:10:16: Applied 23478 samples from profile (offset: 3)
; CHECK: remark: coverage.cc:4:10: Applied 31878 samples from profile (offset: 1)
; CHECK: remark: coverage.cc:11:10: Applied 0 samples from profile (offset: 4)
; CHECK: remark: coverage.cc:10:16: most popular destination for conditional branches at coverage.cc:9:3
;
; There is one sample record with 0 samples at offset 4 in main() that we never
; use:
; CHECK: warning: coverage.cc:7: 4 of 5 available profile records (80%) were applied
;
; Since the unused sample record contributes no samples, sample coverage should
; be 100%. Note that we get this warning because we are requesting an impossible
; 110% coverage check.
; CHECK: warning: coverage.cc:7: 78834 of 78834 available profile samples (100%) were applied

define i64 @_Z3fool(i64 %i) !dbg !4 {
entry:
  %i.addr = alloca i64, align 8
  store i64 %i, i64* %i.addr, align 8
  call void @llvm.dbg.declare(metadata i64* %i.addr, metadata !16, metadata !17), !dbg !18
  %call = call i32 @rand(), !dbg !19
  %conv = sext i32 %call to i64, !dbg !19
  %0 = load i64, i64* %i.addr, align 8, !dbg !20
  %mul = mul nsw i64 %conv, %0, !dbg !21
  ret i64 %mul, !dbg !22
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)

declare i32 @rand()

define i32 @main() !dbg !9 {
entry:
  %retval = alloca i32, align 4
  %sum = alloca i64, align 8
  %i = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata i64* %sum, metadata !23, metadata !17), !dbg !24
  store i64 0, i64* %sum, align 8, !dbg !24
  call void @llvm.dbg.declare(metadata i32* %i, metadata !25, metadata !17), !dbg !27
  store i32 0, i32* %i, align 4, !dbg !27
  br label %for.cond, !dbg !28

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, i32* %i, align 4, !dbg !29
  %cmp = icmp slt i32 %0, 600000000, !dbg !32
  br i1 %cmp, label %for.body, label %for.end, !dbg !33

for.body:                                         ; preds = %for.cond
  %1 = load i32, i32* %i, align 4, !dbg !34
  %conv = sext i32 %1 to i64, !dbg !34
  %call = call i64 @_Z3fool(i64 %conv), !dbg !35
  %2 = load i64, i64* %sum, align 8, !dbg !36
  %add = add nsw i64 %2, %call, !dbg !36
  store i64 %add, i64* %sum, align 8, !dbg !36
  br label %for.inc, !dbg !37

for.inc:                                          ; preds = %for.body
  %3 = load i32, i32* %i, align 4, !dbg !38
  %inc = add nsw i32 %3, 1, !dbg !38
  store i32 %inc, i32* %i, align 4, !dbg !38
  br label %for.cond, !dbg !39

for.end:                                          ; preds = %for.cond
  %4 = load i64, i64* %sum, align 8, !dbg !40
  %cmp1 = icmp sgt i64 %4, 0, !dbg !41
  %cond = select i1 %cmp1, i32 0, i32 1, !dbg !40
  ret i32 %cond, !dbg !42
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 251738) (llvm/trunk 251737)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "coverage.cc", directory: ".")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fool", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !8}
!7 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!8 = !DIBasicType(name: "long int", size: 64, align: 64, encoding: DW_ATE_signed)
!9 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !10, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!12}
!12 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.8.0 (trunk 251738) (llvm/trunk 251737)"}
!16 = !DILocalVariable(name: "i", arg: 1, scope: !4, file: !1, line: 3, type: !8)
!17 = !DIExpression()
!18 = !DILocation(line: 3, column: 24, scope: !4)
!19 = !DILocation(line: 4, column: 10, scope: !4)
!20 = !DILocation(line: 4, column: 19, scope: !4)
!21 = !DILocation(line: 4, column: 17, scope: !4)
!22 = !DILocation(line: 4, column: 3, scope: !4)
!23 = !DILocalVariable(name: "sum", scope: !9, file: !1, line: 8, type: !7)
!24 = !DILocation(line: 8, column: 17, scope: !9)
!25 = !DILocalVariable(name: "i", scope: !26, file: !1, line: 9, type: !12)
!26 = distinct !DILexicalBlock(scope: !9, file: !1, line: 9, column: 3)
!27 = !DILocation(line: 9, column: 12, scope: !26)
!28 = !DILocation(line: 9, column: 8, scope: !26)
!29 = !DILocation(line: 9, column: 19, scope: !30)
!30 = !DILexicalBlockFile(scope: !31, file: !1, discriminator: 2)
!31 = distinct !DILexicalBlock(scope: !26, file: !1, line: 9, column: 3)
!32 = !DILocation(line: 9, column: 21, scope: !30)
!33 = !DILocation(line: 9, column: 3, scope: !30)
!34 = !DILocation(line: 10, column: 16, scope: !31)
!35 = !DILocation(line: 10, column: 12, scope: !31)
!36 = !DILocation(line: 10, column: 9, scope: !31)
!37 = !DILocation(line: 10, column: 5, scope: !31)
!38 = !DILocation(line: 9, column: 39, scope: !31)
!39 = !DILocation(line: 9, column: 3, scope: !31)
!40 = !DILocation(line: 11, column: 10, scope: !9)
!41 = !DILocation(line: 11, column: 14, scope: !9)
!42 = !DILocation(line: 11, column: 3, scope: !9)
