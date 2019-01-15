; RUN: opt %loadPolly -pass-remarks-analysis="polly-scops" -polly-scops \
; RUN: -polly-invariant-load-hoisting=true -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: remark: test/ScopInfo/remarks.c:4:7: SCoP begins here.
; CHECK: remark: test/ScopInfo/remarks.c:9:15: Inbounds assumption:    [N, M, Debug] -> {  : M <= 100 }
; CHECK: remark: test/ScopInfo/remarks.c:13:7: No-error restriction:    [N, M, Debug] -> {  : N > 0 and M >= 0 and (Debug < 0 or Debug > 0) }
; CHECK: remark: test/ScopInfo/remarks.c:8:5: Finite loop restriction:    [N, M, Debug] -> {  : N > 0 and M < 0 }
; CHECK: remark: test/ScopInfo/remarks.c:4:7: No-overflows restriction:    [N, M, Debug] -> {  : M <= -2147483649 - N or M >= 2147483648 - N }
; CHECK: remark: test/ScopInfo/remarks.c:9:18: Possibly aliasing pointer, use restrict keyword.
; CHECK: remark: test/ScopInfo/remarks.c:9:33: Possibly aliasing pointer, use restrict keyword.
; CHECK: remark: test/ScopInfo/remarks.c:9:15: Possibly aliasing pointer, use restrict keyword.
; CHECK: remark: test/ScopInfo/remarks.c:14:3: SCoP ends here.
; CHECK: remark: test/ScopInfo/remarks.c:19:3: SCoP begins here.
; CHECK: remark: test/ScopInfo/remarks.c:21:11: Invariant load restriction:    [tmp] -> {  : tmp < 0 or tmp > 0 }
; CHECK: remark: test/ScopInfo/remarks.c:22:16: SCoP ends here but was dismissed.
;
;    #include <stdio.h>
;
;    void valid(int *A, int *B, int N, int M, int C[100][100], int Debug) {
;      if (N + M == -1)
;        C[0][0] = 0;
;
;      for (int i = 0; i < N; i++) {
;        for (int j = 0; j != M; j++) {
;          C[i][j] += A[i * M + j] + B[i + j];
;        }
;
;        if (Debug)
;          printf("Printf!");
;      }
;    }
;
;    void invalid0(int *A) {
;      for (int i = 0; i < 10; i++)
;        for (int j = 0; j < 10; j++)
;          if (A[0])
;            A[0] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@.str = private unnamed_addr constant [8 x i8] c"Printf!\00", align 1

define void @valid(i32* %A, i32* %B, i32 %N, i32 %M, [100 x i32]* %C, i32 %Debug) #0 !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata i32* %A, i64 0, metadata !23, metadata !24), !dbg !25
  call void @llvm.dbg.value(metadata i32* %B, i64 0, metadata !26, metadata !24), !dbg !27
  call void @llvm.dbg.value(metadata i32 %N, i64 0, metadata !28, metadata !24), !dbg !29
  call void @llvm.dbg.value(metadata i32 %M, i64 0, metadata !30, metadata !24), !dbg !31
  call void @llvm.dbg.value(metadata [100 x i32]* %C, i64 0, metadata !32, metadata !24), !dbg !33
  call void @llvm.dbg.value(metadata i32 %Debug, i64 0, metadata !34, metadata !24), !dbg !35
  br label %entry.split

entry.split:
  %add = add i32 %N, %M, !dbg !36
  %cmp = icmp eq i32 %add, -1, !dbg !38
  br i1 %cmp, label %if.then, label %if.end, !dbg !39

if.then:                                          ; preds = %entry
  %arrayidx1 = getelementptr inbounds [100 x i32], [100 x i32]* %C, i64 0, i64 0, !dbg !40
  store i32 0, i32* %arrayidx1, align 4, !dbg !41
  br label %if.end, !dbg !40

if.end:                                           ; preds = %if.then, %entry
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !42, metadata !24), !dbg !44
  %N64 = sext i32 %N to i64, !dbg !45
  %M64 = sext i32 %M to i64, !dbg !45
  br label %for.cond, !dbg !45

for.cond:                                         ; preds = %for.inc.19, %if.end
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.inc.19 ], [ 0, %if.end ]
  %cmp2 = icmp slt i64 %indvars.iv3, %N64, !dbg !46
  br i1 %cmp2, label %for.body, label %for.end.21, !dbg !49

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !50, metadata !24), !dbg !53
  br label %for.cond.3, !dbg !54

for.cond.3:                                       ; preds = %for.inc, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body ]
  %cmp4 = icmp eq i64 %indvars.iv, %M64, !dbg !55
  br i1 %cmp4, label %for.end, label %for.body.5, !dbg !58

for.body.5:                                       ; preds = %for.cond.3
  %tmp8 = mul i64 %indvars.iv3, %M64, !dbg !59
  %tmp9 = add i64 %tmp8, %indvars.iv, !dbg !61
  %arrayidx7 = getelementptr inbounds i32, i32* %A, i64 %tmp9, !dbg !62
  %tmp10 = load i32, i32* %arrayidx7, align 4, !dbg !62
  %tmp11 = add i64 %indvars.iv3, %indvars.iv, !dbg !63
  %arrayidx10 = getelementptr inbounds i32, i32* %B, i64 %tmp11, !dbg !64
  %tmp12 = load i32, i32* %arrayidx10, align 4, !dbg !64
  %add11 = add i32 %tmp10, %tmp12, !dbg !65
  %arrayidx15 = getelementptr inbounds [100 x i32], [100 x i32]* %C, i64 %indvars.iv3, i64 %indvars.iv, !dbg !66
  %tmp13 = load i32, i32* %arrayidx15, align 4, !dbg !67
  %add16 = add i32 %tmp13, %add11, !dbg !67
  store i32 %add16, i32* %arrayidx15, align 4, !dbg !67
  br label %for.inc, !dbg !68

for.inc:                                          ; preds = %for.body.5
  %indvars.iv.next = add i64 %indvars.iv, 1, !dbg !69
  call void @llvm.dbg.value(metadata !2, i64 0, metadata !50, metadata !24), !dbg !53
  br label %for.cond.3, !dbg !69

for.end:                                          ; preds = %for.cond.3
  %tobool = icmp eq i32 %Debug, 0, !dbg !70
  br i1 %tobool, label %if.end.18, label %if.then.17, !dbg !72

if.then.17:                                       ; preds = %for.end
  %call = call i32 (i8*, ...) @printf(i8* nonnull getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i64 0, i64 0)) #3, !dbg !73
  br label %if.end.18, !dbg !73

if.end.18:                                        ; preds = %for.end, %if.then.17
  br label %for.inc.19, !dbg !74

for.inc.19:                                       ; preds = %if.end.18
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1, !dbg !75
  call void @llvm.dbg.value(metadata !2, i64 0, metadata !42, metadata !24), !dbg !44
  br label %for.cond, !dbg !75

for.end.21:                                       ; preds = %for.cond
  ret void, !dbg !76
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i32 @printf(i8*, ...) #2

define void @invalid0(i32* %A) #0 !dbg !13 {
entry:
  call void @llvm.dbg.value(metadata i32* %A, i64 0, metadata !77, metadata !24), !dbg !78
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !79, metadata !24), !dbg !81
  br label %for.cond, !dbg !82

for.cond:                                         ; preds = %for.inc.5, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc6, %for.inc.5 ]
  %exitcond1 = icmp ne i32 %i.0, 10, !dbg !83
  br i1 %exitcond1, label %for.body, label %for.end.7, !dbg !83

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !86, metadata !24), !dbg !88
  br label %for.cond.1, !dbg !89

for.cond.1:                                       ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %j.0, 10, !dbg !90
  br i1 %exitcond, label %for.body.3, label %for.end, !dbg !90

for.body.3:                                       ; preds = %for.cond.1
  %tmp = load i32, i32* %A, align 4, !dbg !93
  %tobool = icmp eq i32 %tmp, 0, !dbg !93
  br i1 %tobool, label %if.end, label %if.then, !dbg !95

if.then:                                          ; preds = %for.body.3
  store i32 0, i32* %A, align 4, !dbg !96
  br label %if.end, !dbg !97

if.end:                                           ; preds = %for.body.3, %if.then
  br label %for.inc, !dbg !98

for.inc:                                          ; preds = %if.end
  %inc = add nuw nsw i32 %j.0, 1, !dbg !100
  call void @llvm.dbg.value(metadata i32 %inc, i64 0, metadata !86, metadata !24), !dbg !88
  br label %for.cond.1, !dbg !101

for.end:                                          ; preds = %for.cond.1
  br label %for.inc.5, !dbg !102

for.inc.5:                                        ; preds = %for.end
  %inc6 = add nuw nsw i32 %i.0, 1, !dbg !103
  call void @llvm.dbg.value(metadata i32 %inc6, i64 0, metadata !79, metadata !24), !dbg !81
  br label %for.cond, !dbg !104

for.end.7:                                        ; preds = %for.cond
  ret void, !dbg !105
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}
!llvm.ident = !{!22}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0", isOptimized: false, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "test/ScopInfo/remarks.c", directory: "/home/johannes/repos/llvm-polly/tools/polly")
!2 = !{}
!4 = distinct !DISubprogram(name: "valid", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !7, !8, !8, !9, !8}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 3200, align: 32, elements: !11)
!11 = !{!12}
!12 = !DISubrange(count: 100)
!13 = distinct !DISubprogram(name: "invalid0", scope: !1, file: !1, line: 18, type: !14, isLocal: false, isDefinition: true, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !7}
!16 = distinct !DISubprogram(name: "invalid1", scope: !1, file: !1, line: 25, type: !17, isLocal: false, isDefinition: true, scopeLine: 25, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19, !19}
!19 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64)
!20 = !{i32 2, !"Dwarf Version", i32 4}
!21 = !{i32 2, !"Debug Info Version", i32 3}
!22 = !{!"clang version 3.8.0"}
!23 = !DILocalVariable(name: "A", arg: 1, scope: !4, file: !1, line: 3, type: !7)
!24 = !DIExpression()
!25 = !DILocation(line: 3, column: 17, scope: !4)
!26 = !DILocalVariable(name: "B", arg: 2, scope: !4, file: !1, line: 3, type: !7)
!27 = !DILocation(line: 3, column: 25, scope: !4)
!28 = !DILocalVariable(name: "N", arg: 3, scope: !4, file: !1, line: 3, type: !8)
!29 = !DILocation(line: 3, column: 32, scope: !4)
!30 = !DILocalVariable(name: "M", arg: 4, scope: !4, file: !1, line: 3, type: !8)
!31 = !DILocation(line: 3, column: 39, scope: !4)
!32 = !DILocalVariable(name: "C", arg: 5, scope: !4, file: !1, line: 3, type: !9)
!33 = !DILocation(line: 3, column: 46, scope: !4)
!34 = !DILocalVariable(name: "Debug", arg: 6, scope: !4, file: !1, line: 3, type: !8)
!35 = !DILocation(line: 3, column: 63, scope: !4)
!36 = !DILocation(line: 4, column: 9, scope: !37)
!37 = distinct !DILexicalBlock(scope: !4, file: !1, line: 4, column: 7)
!38 = !DILocation(line: 4, column: 13, scope: !37)
!39 = !DILocation(line: 4, column: 7, scope: !4)
!40 = !DILocation(line: 5, column: 5, scope: !37)
!41 = !DILocation(line: 5, column: 13, scope: !37)
!42 = !DILocalVariable(name: "i", scope: !43, file: !1, line: 7, type: !8)
!43 = distinct !DILexicalBlock(scope: !4, file: !1, line: 7, column: 3)
!44 = !DILocation(line: 7, column: 12, scope: !43)
!45 = !DILocation(line: 7, column: 8, scope: !43)
!46 = !DILocation(line: 7, column: 21, scope: !47)
!47 = !DILexicalBlockFile(scope: !48, file: !1, discriminator: 1)
!48 = distinct !DILexicalBlock(scope: !43, file: !1, line: 7, column: 3)
!49 = !DILocation(line: 7, column: 3, scope: !47)
!50 = !DILocalVariable(name: "j", scope: !51, file: !1, line: 8, type: !8)
!51 = distinct !DILexicalBlock(scope: !52, file: !1, line: 8, column: 5)
!52 = distinct !DILexicalBlock(scope: !48, file: !1, line: 7, column: 31)
!53 = !DILocation(line: 8, column: 14, scope: !51)
!54 = !DILocation(line: 8, column: 10, scope: !51)
!55 = !DILocation(line: 8, column: 23, scope: !56)
!56 = !DILexicalBlockFile(scope: !57, file: !1, discriminator: 1)
!57 = distinct !DILexicalBlock(scope: !51, file: !1, line: 8, column: 5)
!58 = !DILocation(line: 8, column: 5, scope: !56)
!59 = !DILocation(line: 9, column: 22, scope: !60)
!60 = distinct !DILexicalBlock(scope: !57, file: !1, line: 8, column: 34)
!61 = !DILocation(line: 9, column: 26, scope: !60)
!62 = !DILocation(line: 9, column: 18, scope: !60)
!63 = !DILocation(line: 9, column: 37, scope: !60)
!64 = !DILocation(line: 9, column: 33, scope: !60)
!65 = !DILocation(line: 9, column: 31, scope: !60)
!66 = !DILocation(line: 9, column: 7, scope: !60)
!67 = !DILocation(line: 9, column: 15, scope: !60)
!68 = !DILocation(line: 10, column: 5, scope: !60)
!69 = !DILocation(line: 8, column: 5, scope: !57)
!70 = !DILocation(line: 12, column: 9, scope: !71)
!71 = distinct !DILexicalBlock(scope: !52, file: !1, line: 12, column: 9)
!72 = !DILocation(line: 12, column: 9, scope: !52)
!73 = !DILocation(line: 13, column: 7, scope: !71)
!74 = !DILocation(line: 14, column: 3, scope: !52)
!75 = !DILocation(line: 7, column: 3, scope: !48)
!76 = !DILocation(line: 16, column: 1, scope: !4)
!77 = !DILocalVariable(name: "A", arg: 1, scope: !13, file: !1, line: 18, type: !7)
!78 = !DILocation(line: 18, column: 20, scope: !13)
!79 = !DILocalVariable(name: "i", scope: !80, file: !1, line: 19, type: !8)
!80 = distinct !DILexicalBlock(scope: !13, file: !1, line: 19, column: 3)
!81 = !DILocation(line: 19, column: 12, scope: !80)
!82 = !DILocation(line: 19, column: 8, scope: !80)
!83 = !DILocation(line: 19, column: 3, scope: !84)
!84 = !DILexicalBlockFile(scope: !85, file: !1, discriminator: 1)
!85 = distinct !DILexicalBlock(scope: !80, file: !1, line: 19, column: 3)
!86 = !DILocalVariable(name: "j", scope: !87, file: !1, line: 20, type: !8)
!87 = distinct !DILexicalBlock(scope: !85, file: !1, line: 20, column: 5)
!88 = !DILocation(line: 20, column: 14, scope: !87)
!89 = !DILocation(line: 20, column: 10, scope: !87)
!90 = !DILocation(line: 20, column: 5, scope: !91)
!91 = !DILexicalBlockFile(scope: !92, file: !1, discriminator: 1)
!92 = distinct !DILexicalBlock(scope: !87, file: !1, line: 20, column: 5)
!93 = !DILocation(line: 21, column: 11, scope: !94)
!94 = distinct !DILexicalBlock(scope: !92, file: !1, line: 21, column: 11)
!95 = !DILocation(line: 21, column: 11, scope: !92)
!96 = !DILocation(line: 22, column: 14, scope: !94)
!97 = !DILocation(line: 22, column: 9, scope: !94)
!98 = !DILocation(line: 21, column: 14, scope: !99)
!99 = !DILexicalBlockFile(scope: !94, file: !1, discriminator: 1)
!100 = !DILocation(line: 20, column: 30, scope: !92)
!101 = !DILocation(line: 20, column: 5, scope: !92)
!102 = !DILocation(line: 22, column: 16, scope: !87)
!103 = !DILocation(line: 19, column: 28, scope: !85)
!104 = !DILocation(line: 19, column: 3, scope: !85)
!105 = !DILocation(line: 23, column: 1, scope: !13)
!106 = !DILocalVariable(name: "A", arg: 1, scope: !16, file: !1, line: 25, type: !19)
!107 = !DILocation(line: 25, column: 21, scope: !16)
!108 = !DILocalVariable(name: "B", arg: 2, scope: !16, file: !1, line: 25, type: !19)
!109 = !DILocation(line: 25, column: 30, scope: !16)
!110 = !DILocalVariable(name: "i", scope: !111, file: !1, line: 26, type: !8)
!111 = distinct !DILexicalBlock(scope: !16, file: !1, line: 26, column: 3)
!112 = !DILocation(line: 26, column: 12, scope: !111)
!113 = !DILocation(line: 26, column: 8, scope: !111)
!114 = !DILocation(line: 26, column: 3, scope: !115)
!115 = !DILexicalBlockFile(scope: !116, file: !1, discriminator: 1)
!116 = distinct !DILexicalBlock(scope: !111, file: !1, line: 26, column: 3)
!117 = !DILocalVariable(name: "j", scope: !118, file: !1, line: 27, type: !8)
!118 = distinct !DILexicalBlock(scope: !116, file: !1, line: 27, column: 5)
!119 = !DILocation(line: 27, column: 14, scope: !118)
!120 = !DILocation(line: 27, column: 10, scope: !118)
!121 = !DILocation(line: 27, column: 5, scope: !122)
!122 = !DILexicalBlockFile(scope: !123, file: !1, discriminator: 1)
!123 = distinct !DILexicalBlock(scope: !118, file: !1, line: 27, column: 5)
!124 = !DILocation(line: 28, column: 17, scope: !123)
!125 = !DILocation(line: 28, column: 7, scope: !123)
!126 = !DILocation(line: 28, column: 15, scope: !123)
!127 = !DILocation(line: 27, column: 5, scope: !123)
!128 = !DILocation(line: 28, column: 23, scope: !118)
!129 = !DILocation(line: 26, column: 3, scope: !116)
!130 = !DILocation(line: 29, column: 1, scope: !16)
