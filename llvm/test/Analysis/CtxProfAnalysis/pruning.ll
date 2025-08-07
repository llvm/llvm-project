; REQUIRES: x86_64-linux
;
; Check that we don't prune the contextual profile, unless the module name
; matches the guid of the root.
;
; RUN: rm -rf %t
; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromYAML --input=%t/profile.yaml --output=%t/profile.ctxprofdata
;
; RUN: cp %t/example.ll %t/1234.ll
; RUN: cp %t/example.ll %t/0x4d2.ll
;
; RUN: opt -passes='require<ctx-prof-analysis>,print<ctx-prof-analysis>' \
; RUN:   -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:   -ctx-profile-printer-level=everything \
; RUN:   %t/example.ll -S 2>&1 | FileCheck %s

; RUN: opt -passes='require<ctx-prof-analysis>,print<ctx-prof-analysis>' \
; RUN:   -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:   -ctx-profile-printer-level=everything \
; RUN:   %t/not-matching.ll -S 2>&1 | FileCheck %s

; RUN: opt -passes='require<ctx-prof-analysis>,print<ctx-prof-analysis>' \
; RUN:   -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:   -ctx-profile-printer-level=everything \
; RUN:   %t/0x4d2.ll -S 2>&1 | FileCheck %s --check-prefix=PRUNED

; CHECK:    Contexts:
; CHECK:      - Guid:            1234
; CHECK:      - Guid:            5678
; CHECK:    FlatProfiles:
; PRUNED-NOT: - Guid:            5678
; PRUNED-NOT: FlatProfiles
;
; pick a large GUID that would be negative, if signed, to test a few ways the
; file name may be formatted.
;--- profile.yaml
Contexts:
  - Guid: 1234
    TotalRootEntryCount: 24
    Counters: [9]
    Callsites:  -
                  - Guid: 1000
                    Counters: [6, 7]

  - Guid: 5678
    TotalRootEntryCount: 24
    Counters: [9]
    Callsites:  -
                  - Guid: 1000
                    Counters: [6, 7]
FlatProfiles:
  - Guid: 777
    Counters: [2]                    
;--- example.ll
define void @an_entrypoint(i32 %a) !guid !0 {
  ret void
}

attributes #0 = { noinline }
!0 = !{ i64  1234 }

;--- not-matching.ll
define void @an_entrypoint(i32 %a) !guid !0 {
  ret void
}

attributes #0 = { noinline }
!0 = !{ i64  1000 }
