; Test that misexpect diagnostics are issued in sample profiling
; RUN: opt < %s -passes="function(lower-expect),sample-profile" -sample-profile-file=%S/Inputs/misexpect.prof -pgo-warn-misexpect -S 2>&1  | FileCheck %s --check-prefix=WARNING

; Test that if expect intrinsics are not lowered, then no diagnostics are issued
; RUN: opt < %s -passes="sample-profile" -sample-profile-file=%S/Inputs/misexpect.prof -pgo-warn-misexpect -S 2>&1  | FileCheck %s --check-prefix=NONE

; Original C++ code for this test case:
;
; #include <stdio.h>
; #include <stdlib.h>

; int main(int argc, char *argv[]) {
;   if (argc < 2)
;     return 1;
;   double result;
;   int limit = atoi(argv[1]);
;   if (limit > 100) {
;     double s = 23.041968 * atoi(argv[2]);
;     for (int u = 0; u < limit; u++) {
;       double x = s;
;       s = x + 3.049 + (double)u;
;       s -= s + 3.94 / x * 0.32;
;     }
;     result = s;
;   } else {
;     result = atoi(argv[2]);
;   }
;   printf("result is %lf\n", result);
;   return 0;
; }

; WARNING-DAG: warning: test.cc:9:14: 20.06%
; WARNING-DAG: warning: test.cc:11:24: 92.74%

; NONE-NOT: warning: test.cc:9:14: 20.06%
; NONE-NOT: warning: test.cc:11:24: 92.74%

@.str = private unnamed_addr constant [15 x i8] c"result is %lf\0A\00", align 1

; Function Attrs: uwtable
define i32 @main(i32 %argc, ptr %argv) #0 !dbg !6 {

entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  %result = alloca double, align 8
  %limit = alloca i32, align 4
  %s = alloca double, align 8
  %u = alloca i32, align 4
  %x = alloca double, align 8
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4
  call void @llvm.dbg.declare(metadata ptr %argc.addr, metadata !16, metadata !17), !dbg !18
  store ptr %argv, ptr %argv.addr, align 8
  call void @llvm.dbg.declare(metadata ptr %argv.addr, metadata !19, metadata !17), !dbg !20
  %0 = load i32, ptr %argc.addr, align 4, !dbg !21
  %cmp = icmp slt i32 %0, 2, !dbg !23
  br i1 %cmp, label %if.then, label %if.end, !dbg !24

if.then:                                          ; preds = %entry
  store i32 1, ptr %retval, align 4, !dbg !25
  br label %return, !dbg !25

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata ptr %result, metadata !26, metadata !17), !dbg !27
  call void @llvm.dbg.declare(metadata ptr %limit, metadata !28, metadata !17), !dbg !29
  %1 = load ptr, ptr %argv.addr, align 8, !dbg !30
  %arrayidx = getelementptr inbounds ptr, ptr %1, i64 1, !dbg !30
  %2 = load ptr, ptr %arrayidx, align 8, !dbg !30
  %call = call i32 @atoi(ptr %2) #4, !dbg !31
  store i32 %call, ptr %limit, align 4, !dbg !29
  %3 = load i32, ptr %limit, align 4, !dbg !32
  %exp = call i32 @llvm.expect.i32(i32 %3, i32 0)
  %tobool = icmp ne i32 %exp, 0, !dbg !34
  br i1 %tobool, label %if.then.2, label %if.else, !dbg !35

if.then.2:                                        ; preds = %if.end
  call void @llvm.dbg.declare(metadata ptr %s, metadata !36, metadata !17), !dbg !38
  %4 = load ptr, ptr %argv.addr, align 8, !dbg !39
  %arrayidx3 = getelementptr inbounds ptr, ptr %4, i64 2, !dbg !39
  %5 = load ptr, ptr %arrayidx3, align 8, !dbg !39
  %call4 = call i32 @atoi(ptr %5) #4, !dbg !40
  %conv = sitofp i32 %call4 to double, !dbg !40
  %mul = fmul double 0x40370ABE6A337A81, %conv, !dbg !41
  store double %mul, ptr %s, align 8, !dbg !38
  call void @llvm.dbg.declare(metadata ptr %u, metadata !42, metadata !17), !dbg !44
  store i32 0, ptr %u, align 4, !dbg !44
  br label %for.cond, !dbg !45

for.cond:                                         ; preds = %for.inc, %if.then.2
  %6 = load i32, ptr %u, align 4, !dbg !46
  %7 = load i32, ptr %limit, align 4, !dbg !48
  %expval = call i32 @llvm.expect.i32(i32 %6, i32 1)
  %cmp5 = icmp ne i32 %expval, 0, !dbg !49
  br i1 %cmp5, label %for.body, label %for.end, !dbg !50

for.body:                                         ; preds = %for.cond
  call void @llvm.dbg.declare(metadata ptr %x, metadata !51, metadata !17), !dbg !53
  %8 = load double, ptr %s, align 8, !dbg !54
  store double %8, ptr %x, align 8, !dbg !53
  %9 = load double, ptr %x, align 8, !dbg !55
  %add = fadd double %9, 3.049000e+00, !dbg !56
  %10 = load i32, ptr %u, align 4, !dbg !57
  %conv6 = sitofp i32 %10 to double, !dbg !57
  %add7 = fadd double %add, %conv6, !dbg !58
  store double %add7, ptr %s, align 8, !dbg !59
  %11 = load double, ptr %s, align 8, !dbg !60
  %12 = load double, ptr %x, align 8, !dbg !61
  %div = fdiv double 3.940000e+00, %12, !dbg !62
  %mul8 = fmul double %div, 3.200000e-01, !dbg !63
  %add9 = fadd double %11, %mul8, !dbg !64
  %13 = load double, ptr %s, align 8, !dbg !65
  %sub = fsub double %13, %add9, !dbg !65
  store double %sub, ptr %s, align 8, !dbg !65
  br label %for.inc, !dbg !66

for.inc:                                          ; preds = %for.body
  %14 = load i32, ptr %u, align 4, !dbg !67
  %inc = add nsw i32 %14, 1, !dbg !67
  store i32 %inc, ptr %u, align 4, !dbg !67
  br label %for.cond, !dbg !68

for.end:                                          ; preds = %for.cond
  %15 = load double, ptr %s, align 8, !dbg !69
  store double %15, ptr %result, align 8, !dbg !70
  br label %if.end.13, !dbg !71

if.else:                                          ; preds = %if.end
  %16 = load ptr, ptr %argv.addr, align 8, !dbg !72
  %arrayidx10 = getelementptr inbounds ptr, ptr %16, i64 2, !dbg !72
  %17 = load ptr, ptr %arrayidx10, align 8, !dbg !72
  %call11 = call i32 @atoi(ptr %17) #4, !dbg !74
  %conv12 = sitofp i32 %call11 to double, !dbg !74
  store double %conv12, ptr %result, align 8, !dbg !75
  br label %if.end.13

if.end.13:                                        ; preds = %if.else, %for.end
  %18 = load double, ptr %result, align 8, !dbg !76
  %call14 = call i32 (ptr, ...) @printf(ptr @.str, double %18), !dbg !77
  store i32 0, ptr %retval, align 4, !dbg !78
  br label %return, !dbg !78

return:                                           ; preds = %if.end.13, %if.then
  %19 = load i32, ptr %retval, align 4, !dbg !79
  ret i32 %19, !dbg !79
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readonly
declare i32 @atoi(ptr) #2

declare i32 @printf(ptr, ...) #3

; Function Attrs: nounwind readnone willreturn
declare i32 @llvm.expect.i32(i32, i32) #5


attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readonly "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readonly }
attributes #5 = { nounwind readnone willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 248211) (llvm/trunk 248217)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "test.cc", directory: "/ssd/llvm_commit")
!2 = !{}
!3 = !{!4}
!4 = !DIBasicType(name: "double", size: 64, align: 64, encoding: DW_ATE_float)
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 4, type: !7, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9, !10}
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 64, align: 64)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, align: 64)
!12 = !DIBasicType(name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.8.0 (trunk 248211) (llvm/trunk 248217)"}
!16 = !DILocalVariable(name: "argc", arg: 1, scope: !6, file: !1, line: 4, type: !9)
!17 = !DIExpression()
!18 = !DILocation(line: 4, column: 15, scope: !6)
!19 = !DILocalVariable(name: "argv", arg: 2, scope: !6, file: !1, line: 4, type: !10)
!20 = !DILocation(line: 4, column: 27, scope: !6)
!21 = !DILocation(line: 5, column: 8, scope: !22)
!22 = distinct !DILexicalBlock(scope: !6, file: !1, line: 5, column: 8)
!23 = !DILocation(line: 5, column: 13, scope: !22)
!24 = !DILocation(line: 5, column: 8, scope: !6)
!25 = !DILocation(line: 6, column: 6, scope: !22)
!26 = !DILocalVariable(name: "result", scope: !6, file: !1, line: 7, type: !4)
!27 = !DILocation(line: 7, column: 11, scope: !6)
!28 = !DILocalVariable(name: "limit", scope: !6, file: !1, line: 8, type: !9)
!29 = !DILocation(line: 8, column: 8, scope: !6)
!30 = !DILocation(line: 8, column: 21, scope: !6)
!31 = !DILocation(line: 8, column: 16, scope: !6)
!32 = !DILocation(line: 9, column: 8, scope: !33)
!33 = distinct !DILexicalBlock(scope: !6, file: !1, line: 9, column: 8)
!34 = !DILocation(line: 9, column: 14, scope: !33)
!35 = !DILocation(line: 9, column: 8, scope: !6)
!36 = !DILocalVariable(name: "s", scope: !37, file: !1, line: 10, type: !4)
!37 = distinct !DILexicalBlock(scope: !33, file: !1, line: 9, column: 21)
!38 = !DILocation(line: 10, column: 13, scope: !37)
!39 = !DILocation(line: 10, column: 34, scope: !37)
!40 = !DILocation(line: 10, column: 29, scope: !37)
!41 = !DILocation(line: 10, column: 27, scope: !37)
!42 = !DILocalVariable(name: "u", scope: !43, file: !1, line: 11, type: !9)
!43 = distinct !DILexicalBlock(scope: !37, file: !1, line: 11, column: 6)
!44 = !DILocation(line: 11, column: 15, scope: !43)
!45 = !DILocation(line: 11, column: 11, scope: !43)
!46 = !DILocation(line: 11, column: 22, scope: !47)
!47 = distinct !DILexicalBlock(scope: !43, file: !1, line: 11, column: 6)
!48 = !DILocation(line: 11, column: 26, scope: !47)
!49 = !DILocation(line: 11, column: 24, scope: !47)
!50 = !DILocation(line: 11, column: 6, scope: !43)
!51 = !DILocalVariable(name: "x", scope: !52, file: !1, line: 12, type: !4)
!52 = distinct !DILexicalBlock(scope: !47, file: !1, line: 11, column: 38)
!53 = !DILocation(line: 12, column: 15, scope: !52)
!54 = !DILocation(line: 12, column: 19, scope: !52)
!55 = !DILocation(line: 13, column: 12, scope: !52)
!56 = !DILocation(line: 13, column: 14, scope: !52)
!57 = !DILocation(line: 13, column: 32, scope: !52)
!58 = !DILocation(line: 13, column: 22, scope: !52)
!59 = !DILocation(line: 13, column: 10, scope: !52)
!60 = !DILocation(line: 14, column: 13, scope: !52)
!61 = !DILocation(line: 14, column: 24, scope: !52)
!62 = !DILocation(line: 14, column: 22, scope: !52)
!63 = !DILocation(line: 14, column: 26, scope: !52)
!64 = !DILocation(line: 14, column: 15, scope: !52)
!65 = !DILocation(line: 14, column: 10, scope: !52)
!66 = !DILocation(line: 15, column: 6, scope: !52)
!67 = !DILocation(line: 11, column: 34, scope: !47)
!68 = !DILocation(line: 11, column: 6, scope: !47)
!69 = !DILocation(line: 16, column: 15, scope: !37)
!70 = !DILocation(line: 16, column: 13, scope: !37)
!71 = !DILocation(line: 17, column: 4, scope: !37)
!72 = !DILocation(line: 18, column: 20, scope: !73)
!73 = distinct !DILexicalBlock(scope: !33, file: !1, line: 17, column: 11)
!74 = !DILocation(line: 18, column: 15, scope: !73)
!75 = !DILocation(line: 18, column: 13, scope: !73)
!76 = !DILocation(line: 20, column: 30, scope: !6)
!77 = !DILocation(line: 20, column: 4, scope: !6)
!78 = !DILocation(line: 21, column: 4, scope: !6)
!79 = !DILocation(line: 22, column: 2, scope: !6)
