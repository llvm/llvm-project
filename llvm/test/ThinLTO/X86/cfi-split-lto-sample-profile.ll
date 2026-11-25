; Verify that internal-linkage CFI functions promoted by ThinLTOBitcodeWriter
; (-thinlto-split-lto-unit) use the standard ".llvm.<md5>" suffix so that after
; LowerTypeTests appends ".cfi", SampleProfileLoader canonicalizes the name and
; matches profiles both when the profile preserves ".__uniq." suffixes
; (SecFlagUniqSuffix) and when it strips them.

; REQUIRES: x86-registered-target

; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-profdata merge --sample --extbinary %t/uniq.proftext -o %t/uniq.afdo
; RUN: llvm-profdata merge --sample --extbinary %t/nouniq.proftext -o %t/nouniq.afdo
; RUN: opt -thinlto-bc -thinlto-split-lto-unit %t/a.ll -o %t/a.bc

; RUN: llvm-lto2 run -save-temps -lto-sample-profile-file=%t/uniq.afdo \
; RUN:   %t/a.bc -o %t/out_uniq \
; RUN:   -r=%t/a.bc,get_callback,plx \
; RUN:   -r=%t/a.bc,dispatch,plx \
; RUN:   -r=%t/a.bc,_ZL8callbacki.__uniq.12345,pl \
; RUN:   -r=%t/a.bc,hot_work, \
; RUN:   -r=%t/a.bc,cold_work,
; RUN: llvm-dis %t/out_uniq.1.4.opt.bc -o - | FileCheck %s

; RUN: llvm-lto2 run -save-temps -use-profiled-call-graph=false \
; RUN:   -lto-sample-profile-file=%t/nouniq.afdo \
; RUN:   %t/a.bc -o %t/out_nouniq \
; RUN:   -r=%t/a.bc,get_callback,plx \
; RUN:   -r=%t/a.bc,dispatch,plx \
; RUN:   -r=%t/a.bc,_ZL8callbacki.__uniq.12345.llvm.23b776f41b961eec8a36ed5b4c89fcd8,pl \
; RUN:   -r=%t/a.bc,hot_work, \
; RUN:   -r=%t/a.bc,cold_work,
; RUN: llvm-dis %t/out_nouniq.1.4.opt.bc -o - | FileCheck %s

; CHECK: define hidden i32 @_ZL8callbacki.__uniq.12345.llvm.23b776f41b961eec8a36ed5b4c89fcd8.cfi(i32 {{.*}}%x){{.*}} !prof ![[ENTRY:[0-9]+]]
; CHECK: br i1 {{.*}}, !prof ![[BW:[0-9]+]]
; CHECK-DAG: ![[ENTRY]] = !{!"function_entry_count", i64 1001}
; CHECK-DAG: ![[BW]] = !{!"branch_weights", i32 1000, i32 2}

;--- uniq.proftext
_ZL8callbacki.__uniq.12345:50000:1000
 1: 1000
 2: 999 hot_work:999
 4: 1 cold_work:1

;--- nouniq.proftext
_ZL8callbacki:50000:1000
 1: 1000
 2: 999 hot_work:999
 4: 1 cold_work:1

;--- a.ll
source_filename = "cfi-split-lto-sample-profile.cc"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local ptr @get_callback() {
  ret ptr @_ZL8callbacki.__uniq.12345
}

define dso_local i32 @dispatch(ptr %fp, i32 %x) {
entry:
  %ok = tail call i1 @llvm.type.test(ptr %fp, metadata !"_ZTSFiiE")
  br i1 %ok, label %cont, label %trap

trap:
  tail call void @llvm.ubsantrap(i8 2)
  unreachable

cont:
  %res = tail call i32 %fp(i32 %x)
  ret i32 %res
}

define internal i32 @_ZL8callbacki.__uniq.12345(i32 %x) #0 !dbg !10 !type !20 {
entry:
  %cmp = icmp sgt i32 %x, 0, !dbg !14
  br i1 %cmp, label %if.then, label %if.else, !dbg !14

if.then:
  %r1 = tail call i32 @hot_work(), !dbg !15
  ret i32 %r1, !dbg !16

if.else:
  %r2 = tail call i32 @cold_work(), !dbg !17
  ret i32 %r2, !dbg !18
}

declare i32 @hot_work()
declare i32 @cold_work()
declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.ubsantrap(i8 immarg)

attributes #0 = { noinline uwtable "sample-profile-suffix-elision-policy"="selected" "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, debugInfoForProfiling: true)
!1 = !DIFile(filename: "cfi-split-lto-sample-profile.cc", directory: ".")
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 5, !"Unique Source File Identifier", !7}
!7 = !{!"cfi-split-lto-sample-profile.cc"}
!10 = distinct !DISubprogram(name: "callback", linkageName: "_ZL8callbacki.__uniq.12345", scope: !1, file: !1, line: 10, type: !11, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{}
!14 = !DILocation(line: 11, column: 7, scope: !10)
!15 = !DILocation(line: 12, column: 10, scope: !10)
!16 = !DILocation(line: 12, column: 3, scope: !10)
!17 = !DILocation(line: 14, column: 10, scope: !10)
!18 = !DILocation(line: 14, column: 3, scope: !10)
!20 = !{i64 0, !"_ZTSFiiE"}
