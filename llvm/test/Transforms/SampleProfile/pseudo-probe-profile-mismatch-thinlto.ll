; REQUIRES: x86_64-linux
; RUN: opt < %S/pseudo-probe-stale-profile-matching-lto.ll -passes='thinlto<O2>' -pgo-kind=pgo-sample-use-pipeline -sample-profile-file=%S/Inputs/pseudo-probe-stale-profile-matching-lto.prof -report-profile-staleness -persist-profile-staleness  -S 2>%t -o %t.ll
; RUN: FileCheck %s --input-file %t
; RUN: FileCheck %s --input-file %t.ll -check-prefix=CHECK-MD

; CHECK: (1/1) of functions' profile are invalid and  (6822/6822) of samples are discarded due to function hash mismatch.
; CHECK: (4/4) of callsites' profile are invalid and (5026/6822) of samples are discarded due to callsite location mismatch.
; CHECK: Out of 4 callsites used for profile matching, 4 callsites have been recovered. After the matching, (0/4) of callsites are still invalid (0/6822) of samples are still discarded.

; CHECK-MD: !{!"NumMismatchedCallsites", i64 4, !"TotalProfiledCallsites", i64 4, !"MismatchedCallsiteSamples", i64 5026, !"TotalProfiledFunc", i64 1, !"TotalFunctionSamples", i64 6822, !"NumMismatchedFuncHash", i64 1, !"MismatchedFuncHashSamples", i64 6822, !"PostMatchNumMismatchedCallsites", i64 0, !"NumCallsitesForMatching", i64 4, !"PostMatchMismatchedCallsiteSamples", i64 0}
