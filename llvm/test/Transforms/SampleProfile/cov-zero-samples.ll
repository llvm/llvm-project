; RUN: opt < %s -instcombine -sample-profile -sample-profile-file=%S/Inputs/cov-zero-samples.prof -sample-profile-check-record-coverage=100 -pass-remarks=sample-profile -o /dev/null 2>&1 | FileCheck %s
; RUN: opt < %s -passes="function(instcombine),sample-profile" -sample-profile-file=%S/Inputs/cov-zero-samples.prof -sample-profile-check-record-coverage=100 -pass-remarks=sample-profile -o /dev/null 2>&1 | FileCheck %s
;
; CHECK: remark: cov-zero-samples.cc:9:29: Applied 404065 samples from profile (offset: 2.1)
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

source_filename = "test/Transforms/SampleProfile/cov-zero-samples.ll"

@N = global i64 8000000000, align 8, !dbg !0
@.str = private unnamed_addr constant [11 x i8] c"sum is %d\0A\00", align 1

define i32 @_Z12never_calledi(i32 %i) !dbg !11 {
entry:
  ret i32 0, !dbg !15
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

define i32 @main() !dbg !17 {
entry:
  %retval = alloca i32, align 4
  %sum = alloca i32, align 4
  %i = alloca i64, align 8
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata i32* %sum, metadata !20, metadata !21), !dbg !22
  store i32 0, i32* %sum, align 4, !dbg !22
  call void @llvm.dbg.declare(metadata i64* %i, metadata !23, metadata !21), !dbg !25
  store i64 0, i64* %i, align 8, !dbg !25
  br label %for.cond, !dbg !26

for.cond:                                         ; preds = %for.inc, %entry

  %0 = load i64, i64* %i, align 8, !dbg !27
  %1 = load volatile i64, i64* @N, align 8, !dbg !30
  %cmp = icmp slt i64 %0, %1, !dbg !31
  br i1 %cmp, label %for.body, label %for.end, !dbg !32

for.body:                                         ; preds = %for.cond
  %2 = load i64, i64* %i, align 8, !dbg !33
  %3 = load volatile i64, i64* @N, align 8, !dbg !36
  %cmp1 = icmp sgt i64 %2, %3, !dbg !37
  br i1 %cmp1, label %if.then, label %if.end, !dbg !38

if.then:                                          ; preds = %for.body
  %4 = load i64, i64* %i, align 8, !dbg !39
  %conv = trunc i64 %4 to i32, !dbg !39
  %call = call i32 @_Z12never_calledi(i32 %conv), !dbg !41
  %5 = load i32, i32* %sum, align 4, !dbg !42
  %add = add nsw i32 %5, %call, !dbg !42
  store i32 %add, i32* %sum, align 4, !dbg !42
  br label %if.end, !dbg !43

if.end:                                           ; preds = %if.then, %for.body
  %6 = load i64, i64* %i, align 8, !dbg !44
  %div = sdiv i64 %6, 239, !dbg !45
  %7 = load i32, i32* %sum, align 4, !dbg !46
  %conv2 = sext i32 %7 to i64, !dbg !46
  %mul = mul nsw i64 %conv2, %div, !dbg !46
  %conv3 = trunc i64 %mul to i32, !dbg !46
  store i32 %conv3, i32* %sum, align 4, !dbg !46
  br label %for.inc, !dbg !47

for.inc:                                          ; preds = %if.end
  %8 = load i64, i64* %i, align 8, !dbg !48
  %inc = add nsw i64 %8, 1, !dbg !48
  store i64 %inc, i64* %i, align 8, !dbg !48
  br label %for.cond, !dbg !50

for.end:                                          ; preds = %for.cond
  %9 = load i32, i32* %sum, align 4, !dbg !51
  %call4 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i32 0, i32 0), i32 %9), !dbg !52
  ret i32 0, !dbg !53
}

declare i32 @printf(i8*, ...)

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = !DIGlobalVariableExpression(var: !1)
!1 = !DIGlobalVariable(name: "N", scope: !2, file: !3, line: 3, type: !6, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 3.8.0 (trunk 253667) (llvm/trunk 253670)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "cov-zero-samples.cc", directory: ".")
!4 = !{}
!5 = !{!0}
!6 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !7)
!7 = !DIBasicType(name: "long long int", size: 64, align: 64, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !{!"clang version 3.8.0 (trunk 253667) (llvm/trunk 253670)"}
!11 = distinct !DISubprogram(name: "never_called", linkageName: "_Z12never_calledi", scope: !3, file: !3, line: 5, type: !12, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!12 = !DISubroutineType(types: !13)
!13 = !{!14, !14}
!14 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 5, column: 27, scope: !16)
!16 = !DILexicalBlockFile(scope: !11, file: !3, discriminator: 3)
!17 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 7, type: !18, isLocal: false, isDefinition: true, scopeLine: 7, flags: DIFlagPrototyped, isOptimized: false, unit: !2, variables: !4)
!18 = !DISubroutineType(types: !19)
!19 = !{!14}
!20 = !DILocalVariable(name: "sum", scope: !17, file: !3, line: 8, type: !14)
!21 = !DIExpression()
!22 = !DILocation(line: 8, column: 7, scope: !17)
!23 = !DILocalVariable(name: "i", scope: !24, file: !3, line: 9, type: !7)
!24 = distinct !DILexicalBlock(scope: !17, file: !3, line: 9, column: 3)
!25 = !DILocation(line: 9, column: 18, scope: !24)
!26 = !DILocation(line: 9, column: 8, scope: !24)
!27 = !DILocation(line: 9, column: 25, scope: !28)
!28 = !DILexicalBlockFile(scope: !29, file: !3, discriminator: 1)
!29 = distinct !DILexicalBlock(scope: !24, file: !3, line: 9, column: 3)
!30 = !DILocation(line: 9, column: 29, scope: !28)
!31 = !DILocation(line: 9, column: 27, scope: !28)
!32 = !DILocation(line: 9, column: 3, scope: !28)
!33 = !DILocation(line: 10, column: 9, scope: !34)
!34 = distinct !DILexicalBlock(scope: !35, file: !3, line: 10, column: 9)
!35 = distinct !DILexicalBlock(scope: !29, file: !3, line: 9, column: 37)
!36 = !DILocation(line: 10, column: 13, scope: !34)
!37 = !DILocation(line: 10, column: 11, scope: !34)
!38 = !DILocation(line: 10, column: 9, scope: !35)
!39 = !DILocation(line: 10, column: 36, scope: !40)
!40 = !DILexicalBlockFile(scope: !34, file: !3, discriminator: 1)
!41 = !DILocation(line: 10, column: 23, scope: !40)
!42 = !DILocation(line: 10, column: 20, scope: !40)
!43 = !DILocation(line: 10, column: 16, scope: !40)
!44 = !DILocation(line: 11, column: 12, scope: !35)
!45 = !DILocation(line: 11, column: 14, scope: !35)
!46 = !DILocation(line: 11, column: 9, scope: !35)
!47 = !DILocation(line: 12, column: 3, scope: !35)
!48 = !DILocation(line: 9, column: 33, scope: !49)
!49 = !DILexicalBlockFile(scope: !29, file: !3, discriminator: 2)
!50 = !DILocation(line: 9, column: 3, scope: !49)
!51 = !DILocation(line: 13, column: 25, scope: !17)
!52 = !DILocation(line: 13, column: 3, scope: !17)
!53 = !DILocation(line: 14, column: 3, scope: !17)

