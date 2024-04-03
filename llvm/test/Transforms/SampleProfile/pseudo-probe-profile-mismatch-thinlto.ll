; REQUIRES: x86_64-linux
; RUN: opt < %S/pseudo-probe-stale-profile-matching-lto.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-matching-lto.prof -report-profile-staleness -persist-profile-staleness  -S 2>%t -o %t.ll
; RUN: FileCheck %s --input-file %t
; RUN: FileCheck %s --input-file %t.ll -check-prefix=CHECK-MD

; CHECK: (1/1) of functions' profile are invalid and  (6822/6822) of samples are discarded due to function hash mismatch.
; CHECK: (4/4) of callsites' profile are invalid and (5026/6822) of samples are discarded due to callsite location mismatch.
; CHECK: (4/4) of callsites and (5026/5026) of samples are recovered by stale profile matching.

; CHECK-MD: ![[#]] = !{!"NumStaleProfileFunc", i64 1, !"TotalProfiledFunc", i64 1, !"MismatchedFunctionSamples", i64 6822, !"TotalFunctionSamples", i64 6822, !"NumMismatchedCallsites", i64 0, !"NumRecoveredCallsites", i64 4, !"TotalProfiledCallsites", i64 4, !"MismatchedCallsiteSamples", i64 0, !"RecoveredCallsiteSamples", i64 5026}
