; REQUIRES: zlib

;; Tests memprof profile matching (with and without instrumentation profiles).

;; TODO: Use text profile inputs once that is available for memprof.

;; The input IR and raw profiles have been generated from the following source:
;;
;; #include <stdlib.h>
;; #include <string.h>
;; #include <unistd.h>
;; char *foo() {
;;   return new char[10];
;; }
;; char *foo2() {
;;   return foo();
;; }
;; char *bar() {
;;   return foo2();
;; }
;; char *baz() {
;;   return foo2();
;; }
;; char *recurse(unsigned n) {
;;   if (!n)
;;     return foo();
;;   return recurse(n-1);
;; }
;; int main(int argc, char **argv) {
;;   // Test allocations with different combinations of stack contexts and
;;   // coldness (based on lifetime, since they are all accessed a single time
;;   // per byte via the memset).
;;   char *a = new char[10];
;;   char *b = new char[10];
;;   char *c = foo();
;;   char *d = foo();
;;   char *e = bar();
;;   char *f = baz();
;;   memset(a, 0, 10);
;;   memset(b, 0, 10);
;;   memset(c, 0, 10);
;;   memset(d, 0, 10);
;;   memset(e, 0, 10);
;;   memset(f, 0, 10);
;;   // a and c have short lifetimes
;;   delete[] a;
;;   delete[] c;
;;   // b, d, e, and f have long lifetimes and will be detected as cold by default.
;;   sleep(200);
;;   delete[] b;
;;   delete[] d;
;;   delete[] e;
;;   delete[] f;
;;   // Loop ensures the two calls to recurse have stack contexts that only differ
;;   // in one level of recursion. We should get two stack contexts reflecting the
;;   // different levels of recursion and different allocation behavior (since the
;;   // first has a very long lifetime and the second has a short lifetime).
;;   for (unsigned i = 0; i < 2; i++) {
;;     char *g = recurse(i + 3);
;;     memset(g, 0, 10);
;;     if (!i)
;;       sleep(200);
;;     delete[] g;
;;   }
;;   return 0;
;; }
;;
;; The following commands were used to compile the source to instrumented
;; executables and collect raw binary format profiles:
;;
;; # Collect memory profile:
;; $ clang++ -fuse-ld=lld -no-pie -Wl,--no-rosegment -gmlt \
;; 	-fdebug-info-for-profiling -mno-omit-leaf-frame-pointer \
;;	-fno-omit-frame-pointer -fno-optimize-sibling-calls -m64 -Wl,-build-id \
;; 	memprof.cc -o memprof.exe -fmemory-profile
;; $ env MEMPROF_OPTIONS=log_path=stdout ./memprof.exe > memprof.memprofraw
;;
;; # Collect IR PGO profile:
;; $ clang++ -fuse-ld=lld -no-pie -Wl,--no-rosegment -gmlt \
;; 	-fdebug-info-for-profiling -mno-omit-leaf-frame-pointer \
;;	-fno-omit-frame-pointer -fno-optimize-sibling-calls -m64 -Wl,-build-id \
;; 	memprof.cc -o pgo.exe -fprofile-generate=.
;; $ ./pgo.exe
;; $ mv default_*.profraw memprof_pgo.profraw
;;
;; # Generate below LLVM IR for use in matching:
;; $ clang++ -gmlt -fdebug-info-for-profiling -fno-omit-frame-pointer \
;;	-fno-optimize-sibling-calls memprof.cc -S -emit-llvm

;; Generate indexed profiles of all combinations:
; RUN: llvm-profdata merge %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe -o %t.memprofdata
; RUN: llvm-profdata merge %S/Inputs/memprof_pgo.profraw %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe -o %t.pgomemprofdata
; RUN: llvm-profdata merge %S/Inputs/memprof_pgo.profraw -o %t.pgoprofdata

;; In all below cases we should not get any messages about missing profile data
;; for any functions. Either we are not performing any matching for a particular
;; profile type or we are performing the matching and it should be successful.
; ALL-NOT: memprof record not found for function hash
; ALL-NOT: no profile data available for function

;; Feed back memprof-only profile
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.memprofdata -pgo-warn-missing-function -S 2>&1 | FileCheck %s --check-prefixes=MEMPROF,ALL,MEMPROFONLY
; There should not be any PGO metadata
; MEMPROFONLY-NOT: !prof

;; Feed back pgo-only profile
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.pgoprofdata -pgo-warn-missing-function -S 2>&1 | FileCheck %s --check-prefixes=PGO,ALL,PGOONLY
; There should not be any memprof related metadata
; PGOONLY-NOT: !memprof
; PGOONLY-NOT: !callsite

;; Feed back pgo+memprof-only profile
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.pgomemprofdata -pgo-warn-missing-function -S 2>&1 | FileCheck %s --check-prefixes=MEMPROF,PGO,ALL

; ModuleID = 'memprof.cc'
source_filename = "memprof.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline optnone uwtable
; ALL-LABEL: define dso_local noundef ptr @_Z3foov()
; There should be some PGO metadata
; PGO: !prof
define dso_local noundef ptr @_Z3foov() #0 !dbg !10 {
entry:
  ; MEMPROF: call {{.*}} @_Znam{{.*}} !memprof ![[M1:[0-9]+]], !callsite ![[C1:[0-9]+]]
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !dbg !13
  ret ptr %call, !dbg !14
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) #1

; Function Attrs: mustprogress noinline optnone uwtable
; ALL-LABEL: define dso_local noundef ptr @_Z4foo2v()
define dso_local noundef ptr @_Z4foo2v() #0 !dbg !15 {
entry:
  ; MEMPROF: call {{.*}} @_Z3foov{{.*}} !callsite ![[C2:[0-9]+]]
  %call = call noundef ptr @_Z3foov(), !dbg !16
  ret ptr %call, !dbg !17
}

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef ptr @_Z3barv() #0 !dbg !18 {
entry:
  ; MEMPROF: call {{.*}} @_Z4foo2v{{.*}} !callsite ![[C3:[0-9]+]]
  %call = call noundef ptr @_Z4foo2v(), !dbg !19
  ret ptr %call, !dbg !20
}

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef ptr @_Z3bazv() #0 !dbg !21 {
entry:
  ; MEMPROF: call {{.*}} @_Z4foo2v{{.*}} !callsite ![[C4:[0-9]+]]
  %call = call noundef ptr @_Z4foo2v(), !dbg !22
  ret ptr %call, !dbg !23
}

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef ptr @_Z7recursej(i32 noundef %n) #0 !dbg !24 {
entry:
  %retval = alloca ptr, align 8
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  %0 = load i32, ptr %n.addr, align 4, !dbg !25
  %tobool = icmp ne i32 %0, 0, !dbg !25
  br i1 %tobool, label %if.end, label %if.then, !dbg !26

if.then:                                          ; preds = %entry
  ; MEMPROF: call {{.*}} @_Z3foov{{.*}} !callsite ![[C5:[0-9]+]]
  %call = call noundef ptr @_Z3foov(), !dbg !27
  store ptr %call, ptr %retval, align 8, !dbg !28
  br label %return, !dbg !28

if.end:                                           ; preds = %entry
  %1 = load i32, ptr %n.addr, align 4, !dbg !29
  %sub = sub i32 %1, 1, !dbg !30
  ; MEMPROF: call {{.*}} @_Z7recursej{{.*}} !callsite ![[C6:[0-9]+]]
  %call1 = call noundef ptr @_Z7recursej(i32 noundef %sub), !dbg !31
  store ptr %call1, ptr %retval, align 8, !dbg !32
  br label %return, !dbg !32

return:                                           ; preds = %if.end, %if.then
  %2 = load ptr, ptr %retval, align 8, !dbg !33
  ret ptr %2, !dbg !33
}

; Function Attrs: mustprogress noinline norecurse optnone uwtable
define dso_local noundef i32 @main(i32 noundef %argc, ptr noundef %argv) #2 !dbg !34 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca ptr, align 8
  %a = alloca ptr, align 8
  %b = alloca ptr, align 8
  %c = alloca ptr, align 8
  %d = alloca ptr, align 8
  %e = alloca ptr, align 8
  %f = alloca ptr, align 8
  %i = alloca i32, align 4
  %g = alloca ptr, align 8
  store i32 0, ptr %retval, align 4
  store i32 %argc, ptr %argc.addr, align 4
  store ptr %argv, ptr %argv.addr, align 8
  ; MEMPROF: call {{.*}} @_Znam{{.*}} #[[A1:[0-9]+]]
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !dbg !35
  store ptr %call, ptr %a, align 8, !dbg !36
  ; MEMPROF: call {{.*}} @_Znam{{.*}} #[[A2:[0-9]+]]
  %call1 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !dbg !37
  store ptr %call1, ptr %b, align 8, !dbg !38
  ; MEMPROF: call {{.*}} @_Z3foov{{.*}} !callsite ![[C7:[0-9]+]]
  %call2 = call noundef ptr @_Z3foov(), !dbg !39
  store ptr %call2, ptr %c, align 8, !dbg !40
  ; MEMPROF: call {{.*}} @_Z3foov{{.*}} !callsite ![[C8:[0-9]+]]
  %call3 = call noundef ptr @_Z3foov(), !dbg !41
  store ptr %call3, ptr %d, align 8, !dbg !42
  ; MEMPROF: call {{.*}} @_Z3barv{{.*}} !callsite ![[C9:[0-9]+]]
  %call4 = call noundef ptr @_Z3barv(), !dbg !43
  store ptr %call4, ptr %e, align 8, !dbg !44
  ; MEMPROF: call {{.*}} @_Z3bazv{{.*}} !callsite ![[C10:[0-9]+]]
  %call5 = call noundef ptr @_Z3bazv(), !dbg !45
  store ptr %call5, ptr %f, align 8, !dbg !46
  %0 = load ptr, ptr %a, align 8, !dbg !47
  call void @llvm.memset.p0.i64(ptr align 1 %0, i8 0, i64 10, i1 false), !dbg !48
  %1 = load ptr, ptr %b, align 8, !dbg !49
  call void @llvm.memset.p0.i64(ptr align 1 %1, i8 0, i64 10, i1 false), !dbg !50
  %2 = load ptr, ptr %c, align 8, !dbg !51
  call void @llvm.memset.p0.i64(ptr align 1 %2, i8 0, i64 10, i1 false), !dbg !52
  %3 = load ptr, ptr %d, align 8, !dbg !53
  call void @llvm.memset.p0.i64(ptr align 1 %3, i8 0, i64 10, i1 false), !dbg !54
  %4 = load ptr, ptr %e, align 8, !dbg !55
  call void @llvm.memset.p0.i64(ptr align 1 %4, i8 0, i64 10, i1 false), !dbg !56
  %5 = load ptr, ptr %f, align 8, !dbg !57
  call void @llvm.memset.p0.i64(ptr align 1 %5, i8 0, i64 10, i1 false), !dbg !58
  %6 = load ptr, ptr %a, align 8, !dbg !59
  %isnull = icmp eq ptr %6, null, !dbg !60
  br i1 %isnull, label %delete.end, label %delete.notnull, !dbg !60

delete.notnull:                                   ; preds = %entry
  call void @_ZdaPv(ptr noundef %6) #7, !dbg !61
  br label %delete.end, !dbg !61

delete.end:                                       ; preds = %delete.notnull, %entry
  %7 = load ptr, ptr %c, align 8, !dbg !63
  %isnull6 = icmp eq ptr %7, null, !dbg !64
  br i1 %isnull6, label %delete.end8, label %delete.notnull7, !dbg !64

delete.notnull7:                                  ; preds = %delete.end
  call void @_ZdaPv(ptr noundef %7) #7, !dbg !65
  br label %delete.end8, !dbg !65

delete.end8:                                      ; preds = %delete.notnull7, %delete.end
  %call9 = call i32 @sleep(i32 noundef 200), !dbg !66
  %8 = load ptr, ptr %b, align 8, !dbg !67
  %isnull10 = icmp eq ptr %8, null, !dbg !68
  br i1 %isnull10, label %delete.end12, label %delete.notnull11, !dbg !68

delete.notnull11:                                 ; preds = %delete.end8
  call void @_ZdaPv(ptr noundef %8) #7, !dbg !69
  br label %delete.end12, !dbg !69

delete.end12:                                     ; preds = %delete.notnull11, %delete.end8
  %9 = load ptr, ptr %d, align 8, !dbg !70
  %isnull13 = icmp eq ptr %9, null, !dbg !71
  br i1 %isnull13, label %delete.end15, label %delete.notnull14, !dbg !71

delete.notnull14:                                 ; preds = %delete.end12
  call void @_ZdaPv(ptr noundef %9) #7, !dbg !72
  br label %delete.end15, !dbg !72

delete.end15:                                     ; preds = %delete.notnull14, %delete.end12
  %10 = load ptr, ptr %e, align 8, !dbg !73
  %isnull16 = icmp eq ptr %10, null, !dbg !74
  br i1 %isnull16, label %delete.end18, label %delete.notnull17, !dbg !74

delete.notnull17:                                 ; preds = %delete.end15
  call void @_ZdaPv(ptr noundef %10) #7, !dbg !75
  br label %delete.end18, !dbg !75

delete.end18:                                     ; preds = %delete.notnull17, %delete.end15
  %11 = load ptr, ptr %f, align 8, !dbg !76
  %isnull19 = icmp eq ptr %11, null, !dbg !77
  br i1 %isnull19, label %delete.end21, label %delete.notnull20, !dbg !77

delete.notnull20:                                 ; preds = %delete.end18
  call void @_ZdaPv(ptr noundef %11) #7, !dbg !78
  br label %delete.end21, !dbg !78

delete.end21:                                     ; preds = %delete.notnull20, %delete.end18
  store i32 0, ptr %i, align 4, !dbg !79
  br label %for.cond, !dbg !80

for.cond:                                         ; preds = %for.inc, %delete.end21
  %12 = load i32, ptr %i, align 4, !dbg !81
  %cmp = icmp ult i32 %12, 2, !dbg !82
  br i1 %cmp, label %for.body, label %for.end, !dbg !83

for.body:                                         ; preds = %for.cond
  %13 = load i32, ptr %i, align 4, !dbg !84
  %add = add i32 %13, 3, !dbg !85
  ; MEMPROF: call {{.*}} @_Z7recursej{{.*}} !callsite ![[C11:[0-9]+]]
  %call22 = call noundef ptr @_Z7recursej(i32 noundef %add), !dbg !86
  store ptr %call22, ptr %g, align 8, !dbg !87
  %14 = load ptr, ptr %g, align 8, !dbg !88
  call void @llvm.memset.p0.i64(ptr align 1 %14, i8 0, i64 10, i1 false), !dbg !89
  %15 = load i32, ptr %i, align 4, !dbg !90
  %tobool = icmp ne i32 %15, 0, !dbg !90
  br i1 %tobool, label %if.end, label %if.then, !dbg !91

if.then:                                          ; preds = %for.body
  %call23 = call i32 @sleep(i32 noundef 200), !dbg !92
  br label %if.end, !dbg !92

if.end:                                           ; preds = %if.then, %for.body
  %16 = load ptr, ptr %g, align 8, !dbg !93
  %isnull24 = icmp eq ptr %16, null, !dbg !94
  br i1 %isnull24, label %delete.end26, label %delete.notnull25, !dbg !94

delete.notnull25:                                 ; preds = %if.end
  call void @_ZdaPv(ptr noundef %16) #7, !dbg !95
  br label %delete.end26, !dbg !95

delete.end26:                                     ; preds = %delete.notnull25, %if.end
  br label %for.inc, !dbg !96

for.inc:                                          ; preds = %delete.end26
  %17 = load i32, ptr %i, align 4, !dbg !97
  %inc = add i32 %17, 1, !dbg !97
  store i32 %inc, ptr %i, align 4, !dbg !97
  br label %for.cond, !dbg !99, !llvm.loop !100

for.end:                                          ; preds = %for.cond
  ret i32 0, !dbg !103
}

; MEMPROF: #[[A1]] = { builtin allocsize(0) "memprof"="notcold" }
; MEMPROF: #[[A2]] = { builtin allocsize(0) "memprof"="cold" }
; MEMPROF: ![[M1]] = !{![[MIB1:[0-9]+]], ![[MIB2:[0-9]+]], ![[MIB3:[0-9]+]], ![[MIB4:[0-9]+]], ![[MIB5:[0-9]+]]}
; MEMPROF: ![[MIB1]] = !{![[STACK1:[0-9]+]], !"notcold"}
; MEMPROF: ![[STACK1]] = !{i64 -2458008693472584243, i64 3952224878458323, i64 -6408471049535768163, i64 -6408471049535768163, i64 -6408471049535768163, i64 -6408471049535768163}
; MEMPROF: ![[MIB2]] = !{![[STACK2:[0-9]+]], !"cold"}
; MEMPROF: ![[STACK2]] = !{i64 -2458008693472584243, i64 3952224878458323, i64 -6408471049535768163, i64 -6408471049535768163, i64 -6408471049535768163, i64 -2523213715586649525}
; MEMPROF: ![[MIB3]] = !{![[STACK3:[0-9]+]], !"cold"}
; MEMPROF: ![[STACK3]] = !{i64 -2458008693472584243, i64 4060711043150162853}
; MEMPROF: ![[MIB4]] = !{![[STACK4:[0-9]+]], !"notcold"}
; MEMPROF: ![[STACK4]] = !{i64 -2458008693472584243, i64 6197270713521362189}
; MEMPROF: ![[MIB5]] = !{![[STACK5:[0-9]+]], !"cold"}
; MEMPROF: ![[STACK5]] = !{i64 -2458008693472584243, i64 -8079659623765193173}
; MEMPROF: ![[C1]] = !{i64 -2458008693472584243}
; MEMPROF: ![[C2]] = !{i64 -8079659623765193173}
; MEMPROF: ![[C3]] = !{i64 -972865200055133905}
; MEMPROF: ![[C4]] = !{i64 -4805294506621015872}
; MEMPROF: ![[C5]] = !{i64 3952224878458323}
; MEMPROF: ![[C6]] = !{i64 -6408471049535768163}
; MEMPROF: ![[C7]] = !{i64 6197270713521362189}
; MEMPROF: ![[C8]] = !{i64 4060711043150162853}
; MEMPROF: ![[C9]] = !{i64 1503792662459039327}
; MEMPROF: ![[C10]] = !{i64 -1910610273966575552}
; MEMPROF: ![[C11]] = !{i64 -2523213715586649525}

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #3

; Function Attrs: nobuiltin nounwind
declare void @_ZdaPv(ptr noundef) #4

declare i32 @sleep(i32 noundef) #5

attributes #0 = { mustprogress noinline optnone uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nobuiltin allocsize(0) "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { mustprogress noinline norecurse optnone uwtable "disable-tail-calls"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { argmemonly nofree nounwind willreturn writeonly }
attributes #4 = { nobuiltin nounwind "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #5 = { "disable-tail-calls"="true" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { builtin allocsize(0) }
attributes #7 = { builtin nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 15.0.0 (https://github.com/llvm/llvm-project.git 6cbe6284d1f0a088b5c6482ae27b738f03d82fe7)", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!1 = !DIFile(filename: "memprof.cc", directory: "/usr/local/google/home/tejohnson/llvm/tmp", checksumkind: CSK_MD5, checksum: "e8c40ebe4b21776b4d60e9632cbc13c2")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 6cbe6284d1f0a088b5c6482ae27b738f03d82fe7)"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 4, type: !11, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!13 = !DILocation(line: 5, column: 10, scope: !10)
!14 = !DILocation(line: 5, column: 3, scope: !10)
!15 = distinct !DISubprogram(name: "foo2", linkageName: "_Z4foo2v", scope: !1, file: !1, line: 7, type: !11, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!16 = !DILocation(line: 8, column: 10, scope: !15)
!17 = !DILocation(line: 8, column: 3, scope: !15)
!18 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !1, file: !1, line: 10, type: !11, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!19 = !DILocation(line: 11, column: 10, scope: !18)
!20 = !DILocation(line: 11, column: 3, scope: !18)
!21 = distinct !DISubprogram(name: "baz", linkageName: "_Z3bazv", scope: !1, file: !1, line: 13, type: !11, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!22 = !DILocation(line: 14, column: 10, scope: !21)
!23 = !DILocation(line: 14, column: 3, scope: !21)
!24 = distinct !DISubprogram(name: "recurse", linkageName: "_Z7recursej", scope: !1, file: !1, line: 16, type: !11, scopeLine: 16, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!25 = !DILocation(line: 17, column: 8, scope: !24)
!26 = !DILocation(line: 17, column: 7, scope: !24)
!27 = !DILocation(line: 18, column: 12, scope: !24)
!28 = !DILocation(line: 18, column: 5, scope: !24)
!29 = !DILocation(line: 19, column: 18, scope: !24)
!30 = !DILocation(line: 19, column: 19, scope: !24)
!31 = !DILocation(line: 19, column: 10, scope: !24)
!32 = !DILocation(line: 19, column: 3, scope: !24)
!33 = !DILocation(line: 20, column: 1, scope: !24)
!34 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 21, type: !11, scopeLine: 21, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !12)
!35 = !DILocation(line: 25, column: 13, scope: !34)
!36 = !DILocation(line: 25, column: 9, scope: !34)
!37 = !DILocation(line: 26, column: 13, scope: !34)
!38 = !DILocation(line: 26, column: 9, scope: !34)
!39 = !DILocation(line: 27, column: 13, scope: !34)
!40 = !DILocation(line: 27, column: 9, scope: !34)
!41 = !DILocation(line: 28, column: 13, scope: !34)
!42 = !DILocation(line: 28, column: 9, scope: !34)
!43 = !DILocation(line: 29, column: 13, scope: !34)
!44 = !DILocation(line: 29, column: 9, scope: !34)
!45 = !DILocation(line: 30, column: 13, scope: !34)
!46 = !DILocation(line: 30, column: 9, scope: !34)
!47 = !DILocation(line: 31, column: 10, scope: !34)
!48 = !DILocation(line: 31, column: 3, scope: !34)
!49 = !DILocation(line: 32, column: 10, scope: !34)
!50 = !DILocation(line: 32, column: 3, scope: !34)
!51 = !DILocation(line: 33, column: 10, scope: !34)
!52 = !DILocation(line: 33, column: 3, scope: !34)
!53 = !DILocation(line: 34, column: 10, scope: !34)
!54 = !DILocation(line: 34, column: 3, scope: !34)
!55 = !DILocation(line: 35, column: 10, scope: !34)
!56 = !DILocation(line: 35, column: 3, scope: !34)
!57 = !DILocation(line: 36, column: 10, scope: !34)
!58 = !DILocation(line: 36, column: 3, scope: !34)
!59 = !DILocation(line: 38, column: 12, scope: !34)
!60 = !DILocation(line: 38, column: 3, scope: !34)
!61 = !DILocation(line: 38, column: 3, scope: !62)
!62 = !DILexicalBlockFile(scope: !34, file: !1, discriminator: 2)
!63 = !DILocation(line: 39, column: 12, scope: !34)
!64 = !DILocation(line: 39, column: 3, scope: !34)
!65 = !DILocation(line: 39, column: 3, scope: !62)
!66 = !DILocation(line: 41, column: 3, scope: !34)
!67 = !DILocation(line: 42, column: 12, scope: !34)
!68 = !DILocation(line: 42, column: 3, scope: !34)
!69 = !DILocation(line: 42, column: 3, scope: !62)
!70 = !DILocation(line: 43, column: 12, scope: !34)
!71 = !DILocation(line: 43, column: 3, scope: !34)
!72 = !DILocation(line: 43, column: 3, scope: !62)
!73 = !DILocation(line: 44, column: 12, scope: !34)
!74 = !DILocation(line: 44, column: 3, scope: !34)
!75 = !DILocation(line: 44, column: 3, scope: !62)
!76 = !DILocation(line: 45, column: 12, scope: !34)
!77 = !DILocation(line: 45, column: 3, scope: !34)
!78 = !DILocation(line: 45, column: 3, scope: !62)
!79 = !DILocation(line: 51, column: 17, scope: !34)
!80 = !DILocation(line: 51, column: 8, scope: !34)
!81 = !DILocation(line: 51, column: 24, scope: !62)
!82 = !DILocation(line: 51, column: 26, scope: !62)
!83 = !DILocation(line: 51, column: 3, scope: !62)
!84 = !DILocation(line: 52, column: 23, scope: !34)
!85 = !DILocation(line: 52, column: 25, scope: !34)
!86 = !DILocation(line: 52, column: 15, scope: !34)
!87 = !DILocation(line: 52, column: 11, scope: !34)
!88 = !DILocation(line: 53, column: 12, scope: !34)
!89 = !DILocation(line: 53, column: 5, scope: !34)
!90 = !DILocation(line: 54, column: 10, scope: !34)
!91 = !DILocation(line: 54, column: 9, scope: !34)
!92 = !DILocation(line: 55, column: 7, scope: !34)
!93 = !DILocation(line: 56, column: 14, scope: !34)
!94 = !DILocation(line: 56, column: 5, scope: !34)
!95 = !DILocation(line: 56, column: 5, scope: !62)
!96 = !DILocation(line: 57, column: 3, scope: !34)
!97 = !DILocation(line: 51, column: 32, scope: !98)
!98 = !DILexicalBlockFile(scope: !34, file: !1, discriminator: 4)
!99 = !DILocation(line: 51, column: 3, scope: !98)
!100 = distinct !{!100, !101, !96, !102}
!101 = !DILocation(line: 51, column: 3, scope: !34)
!102 = !{!"llvm.loop.mustprogress"}
!103 = !DILocation(line: 58, column: 3, scope: !34)
