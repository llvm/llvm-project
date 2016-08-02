; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/cov-zero-samples.prof -sample-profile-check-record-coverage=100 -pass-remarks=sample-profile -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: remark: cov-zero-samples.cc:9:25: Applied 404065 samples from profile (offset: 2.1)
; CHECK: remark: cov-zero-samples.cc:10:9: Applied 443089 samples from profile (offset: 3)
; CHECK: remark: cov-zero-samples.cc:10:36: Applied 0 samples from profile (offset: 3.1)
; CHECK: remark: cov-zero-samples.cc:11:12: Applied 404066 samples from profile (offset: 4)
; CHECK: remark: cov-zero-samples.cc:13:25: Applied 0 samples from profile (offset: 6)
; CHECK: remark: cov-zero-samples.cc:14:3: Applied 0 samples from profile (offset: 7)
; CHECK: remark: cov-zero-samples.cc:10:9: most popular destination for conditional branches at cov-zero-samples.cc:9:3
; CHECK: remark: cov-zero-samples.cc:11:12: most popular destination for conditional branches at cov-zero-samples.cc:10:9
;
; Coverage for this profile should be 100%
; CHECK-NOT: warning: cov-zero-samples.cc:1:

@N = global i64 8000000000, align 8
@.str = private unnamed_addr constant [11 x i8] c"sum is %d\0A\00", align 1

; Function Attrs: nounwind uwtable
define i32 @_Z12never_calledi(i32 %i) !dbg !4 {
entry:
  ret i32 0, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: norecurse uwtable
define i32 @main() !dbg !8 {
entry:
  %retval = alloca i32, align 4
  %sum = alloca i32, align 4
  %i = alloca i64, align 8
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata i32* %sum, metadata !33, metadata !19), !dbg !34
  store i32 0, i32* %sum, align 4, !dbg !34
  call void @llvm.dbg.declare(metadata i64* %i, metadata !35, metadata !19), !dbg !37
  store i64 0, i64* %i, align 8, !dbg !37
  br label %for.cond, !dbg !38

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i64, i64* %i, align 8, !dbg !39
  %1 = load volatile i64, i64* @N, align 8, !dbg !42
  %cmp = icmp slt i64 %0, %1, !dbg !43
  br i1 %cmp, label %for.body, label %for.end, !dbg !44

for.body:                                         ; preds = %for.cond
  %2 = load i64, i64* %i, align 8, !dbg !45
  %3 = load volatile i64, i64* @N, align 8, !dbg !48
  %cmp1 = icmp sgt i64 %2, %3, !dbg !49
  br i1 %cmp1, label %if.then, label %if.end, !dbg !50

if.then:                                          ; preds = %for.body
  %4 = load i64, i64* %i, align 8, !dbg !51
  %conv = trunc i64 %4 to i32, !dbg !51
  %call = call i32 @_Z12never_calledi(i32 %conv), !dbg !53
  %5 = load i32, i32* %sum, align 4, !dbg !54
  %add = add nsw i32 %5, %call, !dbg !54
  store i32 %add, i32* %sum, align 4, !dbg !54
  br label %if.end, !dbg !55

if.end:                                           ; preds = %if.then, %for.body
  %6 = load i64, i64* %i, align 8, !dbg !56
  %div = sdiv i64 %6, 239, !dbg !57
  %7 = load i32, i32* %sum, align 4, !dbg !58
  %conv2 = sext i32 %7 to i64, !dbg !58
  %mul = mul nsw i64 %conv2, %div, !dbg !58
  %conv3 = trunc i64 %mul to i32, !dbg !58
  store i32 %conv3, i32* %sum, align 4, !dbg !58
  br label %for.inc, !dbg !59

for.inc:                                          ; preds = %if.end
  %8 = load i64, i64* %i, align 8, !dbg !60
  %inc = add nsw i64 %8, 1, !dbg !60
  store i64 %inc, i64* %i, align 8, !dbg !60
  br label %for.cond, !dbg !62

for.end:                                          ; preds = %for.cond
  %9 = load i32, i32* %sum, align 4, !dbg !63
  %call4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i32 %9), !dbg !64
  ret i32 0, !dbg !65
}

declare i32 @printf(i8*, ...)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 253667) (llvm/trunk 253670)", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2, subprograms: !3, globals: !11)
!1 = !DIFile(filename: "cov-zero-samples.cc", directory: ".")
!2 = !{}
!3 = !{!4, !8}
!4 = distinct !DISubprogram(name: "never_called", linkageName: "_Z12never_calledi", scope: !1, file: !1, line: 5, type: !5, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 7, type: !9, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, variables: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!7}
!11 = !{!12}
!12 = !DIGlobalVariable(name: "N", scope: !0, file: !1, line: 3, type: !13, isLocal: false, isDefinition: true, variable: i64* @N)
!13 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !14)
!14 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!15 = !{i32 2, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{!"clang version 3.8.0 (trunk 253667) (llvm/trunk 253670)"}
!19 = !DIExpression()
!31 = !DILexicalBlockFile(scope: !4, file: !1, discriminator: 3)
!32 = !DILocation(line: 5, column: 27, scope: !31)
!33 = !DILocalVariable(name: "sum", scope: !8, file: !1, line: 8, type: !7)
!34 = !DILocation(line: 8, column: 7, scope: !8)
!35 = !DILocalVariable(name: "i", scope: !36, file: !1, line: 9, type: !14)
!36 = distinct !DILexicalBlock(scope: !8, file: !1, line: 9, column: 3)
!37 = !DILocation(line: 9, column: 18, scope: !36)
!38 = !DILocation(line: 9, column: 8, scope: !36)
!39 = !DILocation(line: 9, column: 25, scope: !40)
!40 = !DILexicalBlockFile(scope: !41, file: !1, discriminator: 1)
!41 = distinct !DILexicalBlock(scope: !36, file: !1, line: 9, column: 3)
!42 = !DILocation(line: 9, column: 29, scope: !40)
!43 = !DILocation(line: 9, column: 27, scope: !40)
!44 = !DILocation(line: 9, column: 3, scope: !40)
!45 = !DILocation(line: 10, column: 9, scope: !46)
!46 = distinct !DILexicalBlock(scope: !47, file: !1, line: 10, column: 9)
!47 = distinct !DILexicalBlock(scope: !41, file: !1, line: 9, column: 37)
!48 = !DILocation(line: 10, column: 13, scope: !46)
!49 = !DILocation(line: 10, column: 11, scope: !46)
!50 = !DILocation(line: 10, column: 9, scope: !47)
!51 = !DILocation(line: 10, column: 36, scope: !52)
!52 = !DILexicalBlockFile(scope: !46, file: !1, discriminator: 1)
!53 = !DILocation(line: 10, column: 23, scope: !52)
!54 = !DILocation(line: 10, column: 20, scope: !52)
!55 = !DILocation(line: 10, column: 16, scope: !52)
!56 = !DILocation(line: 11, column: 12, scope: !47)
!57 = !DILocation(line: 11, column: 14, scope: !47)
!58 = !DILocation(line: 11, column: 9, scope: !47)
!59 = !DILocation(line: 12, column: 3, scope: !47)
!60 = !DILocation(line: 9, column: 33, scope: !61)
!61 = !DILexicalBlockFile(scope: !41, file: !1, discriminator: 2)
!62 = !DILocation(line: 9, column: 3, scope: !61)
!63 = !DILocation(line: 13, column: 25, scope: !8)
!64 = !DILocation(line: 13, column: 3, scope: !8)
!65 = !DILocation(line: 14, column: 3, scope: !8)
