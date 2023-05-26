; REQUIRES: x86_64-linux
; RUN: opt < %S/profile-mismatch.ll -passes=sample-profile -sample-profile-file=%S/Inputs/profile-mismatch.prof -report-profile-staleness -persist-profile-staleness -flatten-profile-for-matching=1 -S 2>%t -o %t.ll
; RUN: FileCheck %s --input-file %t
; RUN: FileCheck %s --input-file %t.ll -check-prefix=CHECK-MD

; RUN: opt < %S/profile-mismatch.ll -passes=sample-profile -sample-profile-file=%S/Inputs/profile-mismatch-cs.prof -report-profile-staleness -persist-profile-staleness -flatten-profile-for-matching=1 -S 2>%t -o %t.ll
; RUN: FileCheck %s --input-file %t
; RUN: FileCheck %s --input-file %t.ll -check-prefix=CHECK-MD


; CHECK: (3/4) of callsites' profile are invalid and (20/30) of samples are discarded due to callsite location mismatch.

; CHECK-MD: ![[#]] = !{!"NumMismatchedCallsites", i64 3, !"TotalProfiledCallsites", i64 4, !"MismatchedCallsiteSamples", i64 20, !"TotalCallsiteSamples", i64 30}
