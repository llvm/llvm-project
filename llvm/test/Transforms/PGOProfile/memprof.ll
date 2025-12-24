;; Tests memprof profile matching (with and without instrumentation profiles).

;; Several requirements due to using raw profile inputs:
;; PGO profile uses zlib compression
; REQUIRES: zlib
;; Avoid failures on big-endian systems that can't read the profile properly
; REQUIRES: x86_64-linux
;; -stats requires asserts
; REQUIRES: asserts

;; TODO: Use text profile inputs once that is available for memprof.
;; # To update the Inputs below, run Inputs/update_memprof_inputs.sh.
;; # To generate below LLVM IR for use in matching:
;; $ clang++ -gmlt -fdebug-info-for-profiling -fno-omit-frame-pointer \
;;	-fno-optimize-sibling-calls memprof.cc -S -emit-llvm

;; Generate indexed profiles of all combinations:
; RUN: llvm-profdata merge %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe -o %t.memprofdata
; RUN: llvm-profdata merge %S/Inputs/memprof_pgo.proftext %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe -o %t.pgomemprofdata
; RUN: llvm-profdata merge %S/Inputs/memprof_pgo.proftext -o %t.pgoprofdata
; RUN: llvm-profdata merge %S/Inputs/memprof.nocolinfo.memprofraw --profiled-binary %S/Inputs/memprof.nocolinfo.exe -o %t.nocolinfo.memprofdata

;; Check that the summary can be shown (and is identical) for both the raw and indexed profiles.
; RUN: llvm-profdata show --memory %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe | FileCheck %s --check-prefixes=SUMMARY
; RUN: llvm-profdata show --memory %t.memprofdata | FileCheck %s --check-prefixes=SUMMARY
; SUMMARY: # MemProfSummary:
; SUMMARY: #   Total contexts: 8
; SUMMARY: #   Total cold contexts: 5
; SUMMARY: #   Total hot contexts: 0
; SUMMARY: #   Maximum cold context total size: 10
; SUMMARY: #   Maximum warm context total size: 10
; SUMMARY: #   Maximum hot context total size: 0

;; In all below cases we should not get any messages about missing profile data
;; for any functions. Either we are not performing any matching for a particular
;; profile type or we are performing the matching and it should be successful.
; ALL-NOT: memprof record not found for function hash
; ALL-NOT: no profile data available for function

;; Using a memprof-only profile for memprof-use should only give memprof metadata
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -pgo-warn-missing-function -S -memprof-print-match-info -stats 2>&1 | FileCheck %s --check-prefixes=MEMPROF,ALL,MEMPROFONLY,MEMPROFMATCHINFO,MEMPROFSTATS,AMBIG
; There should not be any PGO metadata
; MEMPROFONLY-NOT: !prof

;; Try again but using a profile with missing columns. The memprof matcher
;; should recognize that there are no non-zero columns in the profile and
;; not attempt to include column numbers in the matching (which means that the
;; stack ids will be different).
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.nocolinfo.memprofdata>' -pgo-warn-missing-function -S 2>&1 | FileCheck %s --check-prefixes=MEMPROFNOCOLINFO,ALL,MEMPROFONLY

;; Test the same thing but by passing the memory profile through to a default
;; pipeline via -memory-profile-file=, which should cause the necessary field
;; of the PGOOptions structure to be populated with the profile filename.
; RUN: opt < %s -passes='default<O2>' -memory-profile-file=%t.memprofdata -pgo-warn-missing-function -S 2>&1 | FileCheck %s --check-prefixes=MEMPROF,ALL,MEMPROFONLY,AMBIG

;; Using a pgo+memprof profile for memprof-use should only give memprof metadata
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.pgomemprofdata>' -pgo-warn-missing-function -S 2>&1 | FileCheck %s --check-prefixes=MEMPROF,ALL,MEMPROFONLY,AMBIG

;; Using a pgo-only profile for memprof-use should give an error
; RUN: not opt < %s -passes='memprof-use<profile-filename=%t.pgoprofdata>' -S 2>&1 | FileCheck %s --check-prefixes=MEMPROFWITHPGOONLY
; MEMPROFWITHPGOONLY: Not a memory profile

;; Using a memprof-only profile for pgo-instr-use should give an error
; RUN: not opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.memprofdata -S 2>&1 | FileCheck %s --check-prefixes=PGOWITHMEMPROFONLY
; PGOWITHMEMPROFONLY: Not an IR level instrumentation profile

;; Using a pgo+memprof profile for pgo-instr-use should only give pgo metadata
; RUN: opt < %s -passes=pgo-instr-use -pgo-test-profile-file=%t.pgomemprofdata -pgo-warn-missing-function -S 2>&1 | FileCheck %s --check-prefixes=PGO,ALL,PGOONLY
; There should not be any memprof related metadata
; PGOONLY-NOT: !memprof
; PGOONLY-NOT: !callsite

;; Using a pgo+memprof profile for both memprof-use and pgo-instr-use should
;; give both memprof and pgo metadata.
; RUN: opt < %s -passes='pgo-instr-use,memprof-use<profile-filename=%t.pgomemprofdata>' -pgo-test-profile-file=%t.pgomemprofdata -pgo-warn-missing-function -S 2>&1 | FileCheck %s --check-prefixes=MEMPROF,ALL,PGO,AMBIG

;; Check that the total sizes are reported if requested. A message should be
;; emitted for the pruned context. Also check that remarks are emitted for the
;; allocations hinted without context sensitivity.
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -pgo-warn-missing-function -S -memprof-report-hinted-sizes -pass-remarks=memory-profile-info 2>&1 | FileCheck %s --check-prefixes=TOTALSIZESSINGLE,TOTALSIZES,TOTALSIZENOKEEPALL,REMARKSINGLE

;; Check that the total sizes are reported if requested, and prevent pruning
;; via -memprof-keep-all-not-cold-contexts.
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -pgo-warn-missing-function -S -memprof-report-hinted-sizes -memprof-keep-all-not-cold-contexts 2>&1 | FileCheck %s --check-prefixes=TOTALSIZESSINGLE,TOTALSIZES,TOTALSIZESKEEPALL

;; Check that we hint additional allocations with a threshold < 100%
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -pgo-warn-missing-function -S -memprof-report-hinted-sizes -memprof-matching-cold-threshold=60 2>&1 | FileCheck %s --check-prefixes=TOTALSIZESSINGLE,TOTALSIZESTHRESH60

;; Make sure that the -memprof-cloning-cold-threshold flag is enough to cause
;; the size metadata to be generated for the LTO link.
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -pgo-warn-missing-function -S -memprof-cloning-cold-threshold=80 -memprof-keep-all-not-cold-contexts 2>&1 | FileCheck %s --check-prefixes=TOTALSIZES,TOTALSIZESKEEPALL

;; Make sure we emit a random hotness seed if requested.
; RUN: llvm-profdata merge -memprof-random-hotness %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe -o %t.memprofdatarand 2>&1 | FileCheck %s --check-prefix=RAND
; RAND: random hotness seed =
;; Can't check the exact values, but make sure applying the random profile
;; succeeds with the same stats
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdatarand>' -pgo-warn-missing-function -S -stats 2>&1 | FileCheck %s --check-prefixes=ALL,MEMPROFONLY,MEMPROFSTATS

;; Make sure we use a specific random hotness seed if requested.
; RUN: llvm-profdata merge -memprof-random-hotness -memprof-random-hotness-seed=1730170724 %S/Inputs/memprof.memprofraw --profiled-binary %S/Inputs/memprof.exe -o %t.memprofdatarand2 2>&1 | FileCheck %s --check-prefix=RAND2
; RAND2: random hotness seed = 1730170724
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdatarand2>' -pgo-warn-missing-function -S -stats 2>&1 | FileCheck %s --check-prefixes=MEMPROFRAND2,ALL,MEMPROFONLY,MEMPROFSTATS

;; With the hot access density threshold set to 0, and hot hints enabled,
;; the unconditionally notcold call to new should instead get a hot attribute.
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -pgo-warn-missing-function -S -memprof-print-match-info -stats -memprof-min-ave-lifetime-access-density-hot-threshold=0 -memprof-use-hot-hints 2>&1 | FileCheck %s --check-prefixes=MEMPROFHOT,ALL

;; However, with the same threshold, but hot hints not enabled, it should be
;; notcold again.
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -pgo-warn-missing-function -S -memprof-min-ave-lifetime-access-density-hot-threshold=0 2>&1 | FileCheck %s --check-prefixes=MEMPROF,ALL,AMBIG

;; Test that we don't get an ambiguous memprof attribute when
;; -memprof-ambiguous-attributes is disabled.
; RUN: opt < %s -passes='memprof-use<profile-filename=%t.memprofdata>' -pgo-warn-missing-function -S -memprof-ambiguous-attributes=false 2>&1 | FileCheck %s --check-prefixes=MEMPROF,ALL,NOAMBIG

; MEMPROFMATCHINFO: MemProf notcold context with id 1093248920606587996 has total profiled size 10 is matched with 1 frames
; MEMPROFMATCHINFO: MemProf notcold context with id 5725971306423925017 has total profiled size 10 is matched with 1 frames
; MEMPROFMATCHINFO: MemProf notcold context with id 6792096022461663180 has total profiled size 10 is matched with 1 frames
; MEMPROFMATCHINFO: MemProf cold context with id 8525406123785421946 has total profiled size 10 is matched with 1 frames
; MEMPROFMATCHINFO: MemProf cold context with id 11714230664165068698 has total profiled size 10 is matched with 1 frames
; MEMPROFMATCHINFO: MemProf cold context with id 15737101490731057601 has total profiled size 10 is matched with 1 frames
; MEMPROFMATCHINFO: MemProf cold context with id 16342802530253093571 has total profiled size 10 is matched with 1 frames
; MEMPROFMATCHINFO: MemProf cold context with id 18254812774972004394 has total profiled size 10 is matched with 1 frames
; MEMPROFMATCHINFO: MemProf callsite match for inline call stack 748269490701775343
; MEMPROFMATCHINFO: MemProf callsite match for inline call stack 1544787832369987002
; MEMPROFMATCHINFO: MemProf callsite match for inline call stack 2061451396820446691
; MEMPROFMATCHINFO: MemProf callsite match for inline call stack 2104812325165620841
; MEMPROFMATCHINFO: MemProf callsite match for inline call stack 6281715513834610934
; MEMPROFMATCHINFO: MemProf callsite match for inline call stack 8467819354083268568
; MEMPROFMATCHINFO: MemProf callsite match for inline call stack 8690657650969109624
; MEMPROFMATCHINFO: MemProf callsite match for inline call stack 9086428284934609951
; MEMPROFMATCHINFO: MemProf callsite match for inline call stack 12481870273128938184
; MEMPROFMATCHINFO: MemProf callsite match for inline call stack 12699492813229484831

; ModuleID = 'memprof.cc'
source_filename = "memprof.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline optnone uwtable
; ALL-LABEL: define dso_local noundef{{.*}}ptr @_Z3foov()
; There should be some PGO metadata
; PGO: !prof
define dso_local noundef ptr @_Z3foov() #0 !dbg !10 {
entry:
  ; MEMPROF: call {{.*}} @_Znam{{.*}} #[[A0:[0-9]+]]{{.*}} !memprof ![[M1:[0-9]+]], !callsite ![[C1:[0-9]+]]
  ; MEMPROFNOCOLINFO: call {{.*}} @_Znam{{.*}} !memprof ![[M1:[0-9]+]], !callsite ![[C1:[0-9]+]]
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !dbg !13
  ret ptr %call, !dbg !14
}

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) #1

; Function Attrs: mustprogress noinline optnone uwtable
; ALL-LABEL: define dso_local noundef{{.*}}ptr @_Z4foo2v()
define dso_local noundef ptr @_Z4foo2v() #0 !dbg !15 {
entry:
  ; MEMPROF: call {{.*}} @_Z3foov{{.*}} !callsite ![[C2:[0-9]+]]
  ; MEMPROFNOCOLINFO: call {{.*}} @_Z3foov{{.*}} !callsite ![[C2:[0-9]+]]
  %call = call noundef ptr @_Z3foov(), !dbg !16
  ret ptr %call, !dbg !17
}

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef ptr @_Z3barv() #0 !dbg !18 {
entry:
  ; MEMPROF: call {{.*}} @_Z4foo2v{{.*}} !callsite ![[C3:[0-9]+]]
  ; MEMPROFNOCOLINFO: call {{.*}} @_Z4foo2v{{.*}} !callsite ![[C3:[0-9]+]]
  %call = call noundef ptr @_Z4foo2v(), !dbg !19
  ret ptr %call, !dbg !20
}

; Function Attrs: mustprogress noinline optnone uwtable
define dso_local noundef ptr @_Z3bazv() #0 !dbg !21 {
entry:
  ; MEMPROF: call {{.*}} @_Z4foo2v{{.*}} !callsite ![[C4:[0-9]+]]
  ; MEMPROFNOCOLINFO: call {{.*}} @_Z4foo2v{{.*}} !callsite ![[C4:[0-9]+]]
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
  ; MEMPROFNOCOLINFO: call {{.*}} @_Z3foov{{.*}} !callsite ![[C5:[0-9]+]]
  %call = call noundef ptr @_Z3foov(), !dbg !27
  store ptr %call, ptr %retval, align 8, !dbg !28
  br label %return, !dbg !28

if.end:                                           ; preds = %entry
  %1 = load i32, ptr %n.addr, align 4, !dbg !29
  %sub = sub i32 %1, 1, !dbg !30
  ; MEMPROF: call {{.*}} @_Z7recursej{{.*}} !callsite ![[C6:[0-9]+]]
  ; MEMPROFNOCOLINFO: call {{.*}} @_Z7recursej{{.*}} !callsite ![[C6:[0-9]+]]
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
  ; MEMPROFNOCOLINFO: call {{.*}} @_Znam{{.*}} #[[A1:[0-9]+]]
  ; MEMPROFHOT: call {{.*}} @_Znam{{.*}} #[[A1:[0-9]+]]
  %call = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !dbg !35
  store ptr %call, ptr %a, align 8, !dbg !36
  ; MEMPROF: call {{.*}} @_Znam{{.*}} #[[A2:[0-9]+]]
  ; MEMPROFNOCOLINFO: call {{.*}} @_Znam{{.*}} #[[A2:[0-9]+]]
  %call1 = call noalias noundef nonnull ptr @_Znam(i64 noundef 10) #6, !dbg !37
  store ptr %call1, ptr %b, align 8, !dbg !38
  ; MEMPROF: call {{.*}} @_Z3foov{{.*}} !callsite ![[C7:[0-9]+]]
  ; MEMPROFNOCOLINFO: call {{.*}} @_Z3foov{{.*}} !callsite ![[C7:[0-9]+]]
  %call2 = call noundef ptr @_Z3foov(), !dbg !39
  store ptr %call2, ptr %c, align 8, !dbg !40
  ; MEMPROF: call {{.*}} @_Z3foov{{.*}} !callsite ![[C8:[0-9]+]]
  ; MEMPROFNOCOLINFO: call {{.*}} @_Z3foov{{.*}} !callsite ![[C8:[0-9]+]]
  %call3 = call noundef ptr @_Z3foov(), !dbg !41
  store ptr %call3, ptr %d, align 8, !dbg !42
  ; MEMPROF: call {{.*}} @_Z3barv{{.*}} !callsite ![[C9:[0-9]+]]
  ; MEMPROFNOCOLINFO: call {{.*}} @_Z3barv{{.*}} !callsite ![[C9:[0-9]+]]
  %call4 = call noundef ptr @_Z3barv(), !dbg !43
  store ptr %call4, ptr %e, align 8, !dbg !44
  ; MEMPROF: call {{.*}} @_Z3bazv{{.*}} !callsite ![[C10:[0-9]+]]
  ; MEMPROFNOCOLINFO: call {{.*}} @_Z3bazv{{.*}} !callsite ![[C10:[0-9]+]]
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
  ; MEMPROFNOCOLINFO: call {{.*}} @_Z7recursej{{.*}} !callsite ![[C11:[0-9]+]]
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

;; We optionally apply an ambiguous memprof attribute to ambiguous allocations
; AMBIG: #[[A0]] = { builtin allocsize(0) "memprof"="ambiguous" }
; NOAMBIG: #[[A0]] = { builtin allocsize(0) }
; MEMPROF: #[[A1]] = { builtin allocsize(0) "memprof"="notcold" }
; MEMPROF: #[[A2]] = { builtin allocsize(0) "memprof"="cold" }
; MEMPROF: ![[M1]] = !{![[MIB1:[0-9]+]], ![[MIB2:[0-9]+]], ![[MIB3:[0-9]+]], ![[MIB4:[0-9]+]]}
; MEMPROF: ![[MIB1]] = !{![[STACK1:[0-9]+]], !"cold"}
; MEMPROF: ![[STACK1]] = !{i64 2732490490862098848, i64 748269490701775343}
; MEMPROF: ![[MIB2]] = !{![[STACK2:[0-9]+]], !"cold"}
; MEMPROF: ![[STACK2]] = !{i64 2732490490862098848, i64 2104812325165620841, i64 6281715513834610934, i64 6281715513834610934, i64 6281715513834610934, i64 1544787832369987002}
; MEMPROF: ![[MIB3]] = !{![[STACK3:[0-9]+]], !"notcold"}
; MEMPROF: ![[STACK3]] = !{i64 2732490490862098848, i64 2104812325165620841, i64 6281715513834610934, i64 6281715513834610934, i64 6281715513834610934, i64 6281715513834610934}
; MEMPROF: ![[MIB4]] = !{![[STACK4:[0-9]+]], !"cold"}
; MEMPROF: ![[STACK4]] = !{i64 2732490490862098848, i64 8467819354083268568}
; MEMPROF: ![[C1]] = !{i64 2732490490862098848}
; MEMPROF: ![[C2]] = !{i64 8467819354083268568}
; MEMPROF: ![[C3]] = !{i64 9086428284934609951}
; MEMPROF: ![[C4]] = !{i64 -5964873800580613432}
; MEMPROF: ![[C5]] = !{i64 2104812325165620841}
; MEMPROF: ![[C6]] = !{i64 6281715513834610934}
; MEMPROF: ![[C7]] = !{i64 8690657650969109624}
; MEMPROF: ![[C8]] = !{i64 748269490701775343}
; MEMPROF: ![[C9]] = !{i64 -5747251260480066785}
; MEMPROF: ![[C10]] = !{i64 2061451396820446691}
; MEMPROF: ![[C11]] = !{i64 1544787832369987002}

; TOTALSIZENOKEEPALL: Total size for pruned non-cold full allocation context hash 1093248920606587996: 10

;; For non-context sensitive allocations that get attributes we emit a message
;; with the full allocation context hash, type, and size in bytes.
; TOTALSIZESTHRESH60: Total size for full allocation context hash 8525406123785421946 and dominant alloc type cold: 10
; TOTALSIZESTHRESH60: Total size for full allocation context hash 11714230664165068698 and dominant alloc type cold: 10
; TOTALSIZESTHRESH60: Total size for full allocation context hash 5725971306423925017 and dominant alloc type cold: 10
; TOTALSIZESTHRESH60: Total size for full allocation context hash 16342802530253093571 and dominant alloc type cold: 10
; TOTALSIZESTHRESH60: Total size for full allocation context hash 18254812774972004394 and dominant alloc type cold: 10
; TOTALSIZESTHRESH60: Total size for full allocation context hash 1093248920606587996 and dominant alloc type cold: 10
; TOTALSIZESSINGLE: Total size for full allocation context hash 6792096022461663180 and single alloc type notcold: 10
; REMARKSINGLE: remark: memprof.cc:25:13: call in function main marked with memprof allocation attribute notcold
; TOTALSIZESSINGLE: Total size for full allocation context hash 15737101490731057601 and single alloc type cold: 10
; REMARKSINGLE: remark: memprof.cc:26:13: call in function main marked with memprof allocation attribute cold
;; For context sensitive allocations the full context hash and size in bytes
;; are in separate metadata nodes included on the MIB metadata.
; TOTALSIZES: !"cold", ![[CONTEXT1:[0-9]+]]}
; TOTALSIZES: ![[CONTEXT1]] = !{i64 8525406123785421946, i64 10}
; TOTALSIZES: !"cold", ![[CONTEXT2:[0-9]+]]}
; TOTALSIZES: ![[CONTEXT2]] = !{i64 -6732513409544482918, i64 10}
; TOTALSIZES: !"notcold", ![[CONTEXT3:[0-9]+]]}
; TOTALSIZES: ![[CONTEXT3]] = !{i64 5725971306423925017, i64 10}
;; There can be more than one context id / size pair due to context trimming
;; when we match.
; TOTALSIZES: !"cold", ![[CONTEXT4:[0-9]+]], ![[CONTEXT5:[0-9]+]]}
; TOTALSIZES: ![[CONTEXT4]] = !{i64 -2103941543456458045, i64 10}
; TOTALSIZES: ![[CONTEXT5]] = !{i64 -191931298737547222, i64 10}
; TOTALSIZESKEEPALL: !"notcold", ![[CONTEXT6:[0-9]+]]}
; TOTALSIZESKEEPALL: ![[CONTEXT6]] = !{i64 1093248920606587996, i64 10}

; MEMPROFNOCOLINFO: #[[A1]] = { builtin allocsize(0) "memprof"="notcold" }
; MEMPROFNOCOLINFO: #[[A2]] = { builtin allocsize(0) "memprof"="cold" }
; MEMPROFNOCOLINFO: ![[M1]] = !{![[MIB1:[0-9]+]], ![[MIB2:[0-9]+]], ![[MIB3:[0-9]+]], ![[MIB4:[0-9]+]]}
; MEMPROFNOCOLINFO: ![[MIB1]] = !{![[STACK1:[0-9]+]], !"cold"}
; MEMPROFNOCOLINFO: ![[STACK1]] = !{i64 5281664982037379640, i64 6362220161075421157, i64 -5772587307814069790, i64 -5772587307814069790, i64 -5772587307814069790, i64 3577763375057267810}
; MEMPROFNOCOLINFO: ![[MIB2]] = !{![[STACK2:[0-9]+]], !"notcold"}
; MEMPROFNOCOLINFO: ![[STACK2]] = !{i64 5281664982037379640, i64 6362220161075421157, i64 -5772587307814069790, i64 -5772587307814069790, i64 -5772587307814069790, i64 -5772587307814069790}
; MEMPROFNOCOLINFO: ![[MIB3]] = !{![[STACK3:[0-9]+]], !"cold"}
; MEMPROFNOCOLINFO: ![[STACK3]] = !{i64 5281664982037379640, i64 -6871734214936418908}
; MEMPROFNOCOLINFO: ![[MIB4]] = !{![[STACK4:[0-9]+]], !"cold"}
; MEMPROFNOCOLINFO: ![[STACK4]] = !{i64 5281664982037379640, i64 -6201180255894224618}
; MEMPROFNOCOLINFO: ![[C1]] = !{i64 5281664982037379640}
; MEMPROFNOCOLINFO: ![[C2]] = !{i64 -6871734214936418908}
; MEMPROFNOCOLINFO: ![[C3]] = !{i64 -5588766871448036195}
; MEMPROFNOCOLINFO: ![[C4]] = !{i64 -8990226808646054327}
; MEMPROFNOCOLINFO: ![[C5]] = !{i64 6362220161075421157}
; MEMPROFNOCOLINFO: ![[C6]] = !{i64 -5772587307814069790}
; MEMPROFNOCOLINFO: ![[C7]] = !{i64 -6896091699916449732}
; MEMPROFNOCOLINFO: ![[C8]] = !{i64 -6201180255894224618}
; MEMPROFNOCOLINFO: ![[C9]] = !{i64 -962804290746547393}
; MEMPROFNOCOLINFO: ![[C10]] = !{i64 -4535090212904553409}
; MEMPROFNOCOLINFO: ![[C11]] = !{i64 3577763375057267810}

; MEMPROFHOT: #[[A1]] = { builtin allocsize(0) "memprof"="hot" }

;; For the specific random seed, this is the expected order of hotness
; MEMPROFRAND2: !"cold"
; MEMPROFRAND2: !"cold"
; MEMPROFRAND2: !"cold"
; MEMPROFRAND2: !"notcold"

; MEMPROFSTATS:  8 memprof - Number of alloc contexts in memory profile.
; MEMPROFSTATS: 10 memprof - Number of callsites in memory profile.
; MEMPROFSTATS:  6 memprof - Number of functions having valid memory profile.
; MEMPROFSTATS:  8 memprof - Number of matched memory profile alloc contexts.
; MEMPROFSTATS:  3 memprof - Number of matched memory profile allocs.
; MEMPROFSTATS: 10 memprof - Number of matched memory profile callsites.


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
