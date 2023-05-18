; Test that misexpect diagnostics are emtted when profile branch weights are
; below the branch weights added by llvm.expect intrinsics

; RUN: llvm-profdata merge %S/Inputs/misexpect-branch.proftext -o %t.profdata

; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -S 2>&1 | FileCheck %s --check-prefix=WARNING
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pass-remarks=misexpect -S 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -pass-remarks=misexpect -S 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -S 2>&1 | FileCheck %s --check-prefix=DISABLED
; RUN: opt < %s -passes="function(lower-expect),pgo-instr-use" -pgo-test-profile-file=%t.profdata -pgo-warn-misexpect -misexpect-tolerance=81 -pass-remarks=misexpect -S 2>&1 | FileCheck %s --check-prefix=THRESHOLD

; WARNING-DAG: warning: misexpect-branch.c:22:0: 19.98%
; WARNING-NOT: remark: misexpect-branch.c:22:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 19.98% (399668 / 2000000) of profiled executions.

; REMARK-NOT: warning: misexpect-branch.c:22:0: 19.98%
; REMARK-DAG: remark: misexpect-branch.c:22:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 19.98% (399668 / 2000000) of profiled executions.

; BOTH-DAG: warning: misexpect-branch.c:22:0: 19.98%
; BOTH-DAG: remark: misexpect-branch.c:22:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 19.98% (399668 / 2000000) of profiled executions.

; DISABLED-NOT: warning: misexpect-branch.c:22:0: 19.98%
; DISABLED-NOT: remark: misexpect-branch.c:22:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 19.98% (399668 / 2000000) of profiled executions.

; THRESHOLD-NOT: warning: misexpect-branch.c:22:0: 19.98%
; THRESHOLD-NOT: remark: misexpect-branch.c:22:0: Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on 19.98% (399668 / 2000000) of profiled executions.

; ModuleID = 'misexpect-branch.c'
source_filename = "misexpect-branch.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@inner_loop = constant i32 100, align 4
@outer_loop = constant i32 2000, align 4

; Function Attrs: nounwind
define i32 @bar() #0 !dbg !6 {
entry:
  %rando = alloca i32, align 4
  %x = alloca i32, align 4
  call void @llvm.lifetime.start.p0(i64 4, ptr %rando) #4, !dbg !9
  %call = call i32 (...) @buzz(), !dbg !9
  store i32 %call, ptr %rando, align 4, !dbg !9, !tbaa !10
  call void @llvm.lifetime.start.p0(i64 4, ptr %x) #4, !dbg !14
  store i32 0, ptr %x, align 4, !dbg !14, !tbaa !10
  %0 = load i32, ptr %rando, align 4, !dbg !15, !tbaa !10
  %rem = srem i32 %0, 200000, !dbg !15
  %cmp = icmp eq i32 %rem, 0, !dbg !15
  %lnot = xor i1 %cmp, true, !dbg !15
  %lnot1 = xor i1 %lnot, true, !dbg !15
  %lnot.ext = zext i1 %lnot1 to i32, !dbg !15
  %conv = sext i32 %lnot.ext to i64, !dbg !15
  %expval = call i64 @llvm.expect.i64(i64 %conv, i64 1), !dbg !15
  %tobool = icmp ne i64 %expval, 0, !dbg !15
  br i1 %tobool, label %if.then, label %if.else, !dbg !15

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %rando, align 4, !dbg !16, !tbaa !10
  %call2 = call i32 @baz(i32 %1), !dbg !16
  store i32 %call2, ptr %x, align 4, !dbg !16, !tbaa !10
  br label %if.end, !dbg !17

if.else:                                          ; preds = %entry
  %call3 = call i32 @foo(i32 50), !dbg !18
  store i32 %call3, ptr %x, align 4, !dbg !18, !tbaa !10
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %2 = load i32, ptr %x, align 4, !dbg !19, !tbaa !10
  call void @llvm.lifetime.end.p0(i64 4, ptr %x) #4, !dbg !20
  call void @llvm.lifetime.end.p0(i64 4, ptr %rando) #4, !dbg !20
  ret i32 %2, !dbg !19
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

declare i32 @buzz(...) #2

; Function Attrs: nounwind readnone willreturn
declare i64 @llvm.expect.i64(i64, i64) #3

declare i32 @baz(i32) #2

declare i32 @foo(i32) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone willreturn }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (trunk c20270bfffc9d6965219de339d66c61e9fe7d82d)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: ".")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 10.0.0 (trunk c20270bfffc9d6965219de339d66c61e9fe7d82d)"}
!6 = distinct !DISubprogram(name: "bar", scope: !7, file: !7, line: 19, type: !8, scopeLine: 19, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!7 = !DIFile(filename: "misexpect-branch.c", directory: ".")
!8 = !DISubroutineType(types: !2)
!9 = !DILocation(line: 20, scope: !6)
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
!14 = !DILocation(line: 21, scope: !6)
!15 = !DILocation(line: 22, scope: !6)
!16 = !DILocation(line: 23, scope: !6)
!17 = !DILocation(line: 24, scope: !6)
!18 = !DILocation(line: 25, scope: !6)
!19 = !DILocation(line: 27, scope: !6)
!20 = !DILocation(line: 28, scope: !6)
