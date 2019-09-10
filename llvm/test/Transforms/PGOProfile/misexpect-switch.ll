
; RUN: llvm-profdata merge %S/Inputs/misexpect-switch.proftext -o %t.profdata
; RUN: llvm-profdata merge %S/Inputs/misexpect-switch-correct.proftext -o %t.c.profdata

; RUN: opt < %s -lower-expect -pgo-instr-use -pgo-test-profile-file=%t.profdata -S -pgo-warn-misexpect 2>&1 | FileCheck %s --check-prefix=WARNING
; RUN: opt < %s -lower-expect -pgo-instr-use -pgo-test-profile-file=%t.profdata -S -pass-remarks=misexpect 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt < %s -lower-expect -pgo-instr-use -pgo-test-profile-file=%t.profdata -S -pgo-warn-misexpect -pass-remarks=misexpect 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: opt < %s -lower-expect -pgo-instr-use -pgo-test-profile-file=%t.profdata -S 2>&1 | FileCheck %s --check-prefix=DISABLED

; New PM
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -S 2>&1 | FileCheck %s --check-prefix=WARNING
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pass-remarks=misexpect -S 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -pass-remarks=misexpect -S 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -S 2>&1 | FileCheck %s --check-prefix=DISABLED

; RUN: opt < %s -lower-expect -pgo-instr-use -pgo-test-profile-file=%t.c.profdata -S -pgo-warn-misexpect -pass-remarks=misexpect 2>&1 | FileCheck %s --check-prefix=CORRECT
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.c.profdata -pgo-warn-misexpect -pass-remarks=misexpect -S 2>&1 | FileCheck %s --check-prefix=CORRECT

; WARNING-DAG: warning: misexpect-switch.c:26:5: 0.00%
; WARNING-NOT: remark: misexpect-switch.c:26:5: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 0.00% (0 / 8112) of profiled executions.

; REMARK-NOT: warning: misexpect-switch.c:26:5: 0.00%
; REMARK-DAG: remark: misexpect-switch.c:26:5: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 0.00% (0 / 8112) of profiled executions.

; BOTH-DAG: warning: misexpect-switch.c:26:5: 0.00%
; BOTH-DAG: remark: misexpect-switch.c:26:5: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 0.00% (0 / 8112) of profiled executions.

; DISABLED-NOT: warning: misexpect-switch.c:26:5: 0.00%
; DISABLED-NOT: remark: misexpect-switch.c:26:5: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 0.00% (0 / 8112) of profiled executions.

; DISABLED-NOT: warning: misexpect-switch.c:26:5: 0.00%
; DISABLED-NOT: remark: misexpect-switch.c:26:5: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 0.00% (0 / 8112) of profiled executions.

; CORRECT-NOT: warning: {{.*}}
; CORRECT-NOT: remark: {{.*}}
; CHECK-DAG: !{!"misexpect", i64 0, i64 2000, i64 1}



; ModuleID = 'misexpect-switch.c'
source_filename = "misexpect-switch.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@inner_loop = dso_local constant i32 1000, align 4, !dbg !0
@outer_loop = dso_local constant i32 20, align 4, !dbg !6
@arry_size = dso_local constant i32 25, align 4, !dbg !10
@arry = dso_local global [25 x i32] zeroinitializer, align 16, !dbg !12

; Function Attrs: nounwind uwtable
define dso_local void @init_arry() #0 !dbg !21 {
entry:
  %i = alloca i32, align 4
  %0 = bitcast i32* %i to i8*, !dbg !26
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #6, !dbg !26
  call void @llvm.dbg.declare(metadata i32* %i, metadata !25, metadata !DIExpression()), !dbg !27
  store i32 0, i32* %i, align 4, !dbg !28, !tbaa !30
  br label %for.cond, !dbg !34

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4, !dbg !35, !tbaa !30
  %cmp = icmp slt i32 %1, 25, !dbg !37
  br i1 %cmp, label %for.body, label %for.end, !dbg !38

for.body:                                         ; preds = %for.cond
  %call = call i32 @rand() #6, !dbg !39
  %rem = srem i32 %call, 10, !dbg !41
  %2 = load i32, i32* %i, align 4, !dbg !42, !tbaa !30
  %idxprom = sext i32 %2 to i64, !dbg !43
  %arrayidx = getelementptr inbounds [25 x i32], [25 x i32]* @arry, i64 0, i64 %idxprom, !dbg !43
  store i32 %rem, i32* %arrayidx, align 4, !dbg !44, !tbaa !30
  br label %for.inc, !dbg !45

for.inc:                                          ; preds = %for.body
  %3 = load i32, i32* %i, align 4, !dbg !46, !tbaa !30
  %inc = add nsw i32 %3, 1, !dbg !46
  store i32 %inc, i32* %i, align 4, !dbg !46, !tbaa !30
  br label %for.cond, !dbg !47, !llvm.loop !48

for.end:                                          ; preds = %for.cond
  %4 = bitcast i32* %i to i8*, !dbg !50
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #6, !dbg !50
  ret void, !dbg !50
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nounwind
declare dso_local i32 @rand() #3

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main() #0 !dbg !51 {
entry:
  %retval = alloca i32, align 4
  %val = alloca i32, align 4
  %j = alloca i32, align 4
  %condition = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  call void @init_arry(), !dbg !62
  %0 = bitcast i32* %val to i8*, !dbg !63
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #6, !dbg !63
  call void @llvm.dbg.declare(metadata i32* %val, metadata !55, metadata !DIExpression()), !dbg !64
  store i32 0, i32* %val, align 4, !dbg !64, !tbaa !30
  %1 = bitcast i32* %j to i8*, !dbg !65
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1) #6, !dbg !65
  call void @llvm.dbg.declare(metadata i32* %j, metadata !56, metadata !DIExpression()), !dbg !66
  store i32 0, i32* %j, align 4, !dbg !67, !tbaa !30
  br label %for.cond, !dbg !68

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, i32* %j, align 4, !dbg !69, !tbaa !30
  %cmp = icmp slt i32 %2, 20000, !dbg !70
  br i1 %cmp, label %for.body, label %for.end, !dbg !71

for.body:                                         ; preds = %for.cond
  %3 = bitcast i32* %condition to i8*, !dbg !72
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %3) #6, !dbg !72
  call void @llvm.dbg.declare(metadata i32* %condition, metadata !57, metadata !DIExpression()), !dbg !73
  %call = call i32 @rand() #6, !dbg !74
  %rem = srem i32 %call, 5, !dbg !75
  store i32 %rem, i32* %condition, align 4, !dbg !73, !tbaa !30
  %4 = load i32, i32* %condition, align 4, !dbg !76, !tbaa !30
  %conv = zext i32 %4 to i64, !dbg !76
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 0), !dbg !77
  switch i64 %expval, label %sw.default [
    i64 0, label %sw.bb
    i64 1, label %sw.bb2
    i64 2, label %sw.bb2
    i64 3, label %sw.bb2
    i64 4, label %sw.bb3
  ], !dbg !78

sw.bb:                                            ; preds = %for.body
  %call1 = call i32 @sum(i32* getelementptr inbounds ([25 x i32], [25 x i32]* @arry, i64 0, i64 0), i32 25), !dbg !79
  %5 = load i32, i32* %val, align 4, !dbg !81, !tbaa !30
  %add = add nsw i32 %5, %call1, !dbg !81
  store i32 %add, i32* %val, align 4, !dbg !81, !tbaa !30
  br label %sw.epilog, !dbg !82

sw.bb2:                                           ; preds = %for.body, %for.body, %for.body
  br label %sw.epilog, !dbg !83

sw.bb3:                                           ; preds = %for.body
  %call4 = call i32 @random_sample(i32* getelementptr inbounds ([25 x i32], [25 x i32]* @arry, i64 0, i64 0), i32 25), !dbg !84
  %6 = load i32, i32* %val, align 4, !dbg !85, !tbaa !30
  %add5 = add nsw i32 %6, %call4, !dbg !85
  store i32 %add5, i32* %val, align 4, !dbg !85, !tbaa !30
  br label %sw.epilog, !dbg !86

sw.default:                                       ; preds = %for.body
  unreachable, !dbg !87

sw.epilog:                                        ; preds = %sw.bb3, %sw.bb2, %sw.bb
  %7 = bitcast i32* %condition to i8*, !dbg !88
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %7) #6, !dbg !88
  br label %for.inc, !dbg !89

for.inc:                                          ; preds = %sw.epilog
  %8 = load i32, i32* %j, align 4, !dbg !90, !tbaa !30
  %inc = add nsw i32 %8, 1, !dbg !90
  store i32 %inc, i32* %j, align 4, !dbg !90, !tbaa !30
  br label %for.cond, !dbg !91, !llvm.loop !92

for.end:                                          ; preds = %for.cond
  %9 = bitcast i32* %j to i8*, !dbg !94
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %9) #6, !dbg !94
  %10 = bitcast i32* %val to i8*, !dbg !94
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %10) #6, !dbg !94
  ret i32 0, !dbg !95
}

; Function Attrs: nounwind readnone willreturn
declare i64 @llvm.expect.i64(i64, i64) #4

declare dso_local i32 @sum(i32*, i32) #5

declare dso_local i32 @random_sample(i32*, i32) #5

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind readnone speculatable willreturn }
attributes #3 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone willreturn }
attributes #5 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "inner_loop", scope: !2, file: !3, line: 7, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, nameTableKind: None)
!3 = !DIFile(filename: "misexpect-switch.c", directory: ".")
!4 = !{}
!5 = !{!0, !6, !10, !12}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "outer_loop", scope: !2, file: !3, line: 8, type: !8, isLocal: false, isDefinition: true)
!8 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "arry_size", scope: !2, file: !3, line: 9, type: !8, isLocal: false, isDefinition: true)
!12 = !DIGlobalVariableExpression(var: !13, expr: !DIExpression())
!13 = distinct !DIGlobalVariable(name: "arry", scope: !2, file: !3, line: 11, type: !14, isLocal: false, isDefinition: true)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 800, elements: !15)
!15 = !{!16}
!16 = !DISubrange(count: 25)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 1, !"wchar_size", i32 4}
!20 = !{!"clang version 10.0.0"}
!21 = distinct !DISubprogram(name: "init_arry", scope: !3, file: !3, line: 13, type: !22, scopeLine: 13, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !24)
!22 = !DISubroutineType(types: !23)
!23 = !{null}
!24 = !{!25}
!25 = !DILocalVariable(name: "i", scope: !21, file: !3, line: 14, type: !9)
!26 = !DILocation(line: 14, column: 3, scope: !21)
!27 = !DILocation(line: 14, column: 7, scope: !21)
!28 = !DILocation(line: 15, column: 10, scope: !29)
!29 = distinct !DILexicalBlock(scope: !21, file: !3, line: 15, column: 3)
!30 = !{!31, !31, i64 0}
!31 = !{!"int", !32, i64 0}
!32 = !{!"omnipotent char", !33, i64 0}
!33 = !{!"Simple C/C++ TBAA"}
!34 = !DILocation(line: 15, column: 8, scope: !29)
!35 = !DILocation(line: 15, column: 15, scope: !36)
!36 = distinct !DILexicalBlock(scope: !29, file: !3, line: 15, column: 3)
!37 = !DILocation(line: 15, column: 17, scope: !36)
!38 = !DILocation(line: 15, column: 3, scope: !29)
!39 = !DILocation(line: 16, column: 15, scope: !40)
!40 = distinct !DILexicalBlock(scope: !36, file: !3, line: 15, column: 35)
!41 = !DILocation(line: 16, column: 22, scope: !40)
!42 = !DILocation(line: 16, column: 10, scope: !40)
!43 = !DILocation(line: 16, column: 5, scope: !40)
!44 = !DILocation(line: 16, column: 13, scope: !40)
!45 = !DILocation(line: 17, column: 3, scope: !40)
!46 = !DILocation(line: 15, column: 30, scope: !36)
!47 = !DILocation(line: 15, column: 3, scope: !36)
!48 = distinct !{!48, !38, !49}
!49 = !DILocation(line: 17, column: 3, scope: !29)
!50 = !DILocation(line: 18, column: 1, scope: !21)
!51 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 20, type: !52, scopeLine: 20, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !54)
!52 = !DISubroutineType(types: !53)
!53 = !{!9}
!54 = !{!55, !56, !57}
!55 = !DILocalVariable(name: "val", scope: !51, file: !3, line: 22, type: !9)
!56 = !DILocalVariable(name: "j", scope: !51, file: !3, line: 23, type: !9)
!57 = !DILocalVariable(name: "condition", scope: !58, file: !3, line: 25, type: !61)
!58 = distinct !DILexicalBlock(scope: !59, file: !3, line: 24, column: 49)
!59 = distinct !DILexicalBlock(scope: !60, file: !3, line: 24, column: 3)
!60 = distinct !DILexicalBlock(scope: !51, file: !3, line: 24, column: 3)
!61 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!62 = !DILocation(line: 21, column: 3, scope: !51)
!63 = !DILocation(line: 22, column: 3, scope: !51)
!64 = !DILocation(line: 22, column: 7, scope: !51)
!65 = !DILocation(line: 23, column: 3, scope: !51)
!66 = !DILocation(line: 23, column: 7, scope: !51)
!67 = !DILocation(line: 24, column: 10, scope: !60)
!68 = !DILocation(line: 24, column: 8, scope: !60)
!69 = !DILocation(line: 24, column: 15, scope: !59)
!70 = !DILocation(line: 24, column: 17, scope: !59)
!71 = !DILocation(line: 24, column: 3, scope: !60)
!72 = !DILocation(line: 25, column: 5, scope: !58)
!73 = !DILocation(line: 25, column: 14, scope: !58)
!74 = !DILocation(line: 25, column: 26, scope: !58)
!75 = !DILocation(line: 25, column: 33, scope: !58)
!76 = !DILocation(line: 26, column: 30, scope: !58)
!77 = !DILocation(line: 26, column: 13, scope: !58)
!78 = !DILocation(line: 26, column: 5, scope: !58)
!79 = !DILocation(line: 28, column: 14, scope: !80)
!80 = distinct !DILexicalBlock(scope: !58, file: !3, line: 26, column: 45)
!81 = !DILocation(line: 28, column: 11, scope: !80)
!82 = !DILocation(line: 29, column: 7, scope: !80)
!83 = !DILocation(line: 33, column: 7, scope: !80)
!84 = !DILocation(line: 35, column: 14, scope: !80)
!85 = !DILocation(line: 35, column: 11, scope: !80)
!86 = !DILocation(line: 36, column: 7, scope: !80)
!87 = !DILocation(line: 38, column: 7, scope: !80)
!88 = !DILocation(line: 40, column: 3, scope: !59)
!89 = !DILocation(line: 40, column: 3, scope: !58)
!90 = !DILocation(line: 24, column: 44, scope: !59)
!91 = !DILocation(line: 24, column: 3, scope: !59)
!92 = distinct !{!92, !71, !93}
!93 = !DILocation(line: 40, column: 3, scope: !60)
!94 = !DILocation(line: 43, column: 1, scope: !51)
!95 = !DILocation(line: 42, column: 3, scope: !51)
