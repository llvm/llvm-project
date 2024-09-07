; REQUIRES: x86_64-linux
;
; Check that, if none of the roots in the profile are defined in the module, the
; profile is treated as empty (i.e. "none provided")
;
; RUN: rm -rf %t
; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromJSON --input=%t/profile.json --output=%t/profile.ctxprofdata
; RUN: opt -passes='require<ctx-prof-analysis>,print<ctx-prof-analysis>' -ctx-profile-printer-level=everything \
; RUN:   %t/example.ll -S 2>&1 | FileCheck %s

; CHECK: No contextual profile was provided
;
; This is the reference profile, laid out in the format the json formatter will
; output it from opt.
;--- profile.json
[
  {
    "Counters": [
      9
    ],
    "Guid": 12341
  },
  {
    "Counters": [
      5
    ],
    "Guid": 12074870348631550642
  },
  {
    "Callsites": [
      [
        {
          "Counters": [
            6,
            7
          ],
          "Guid": 728453322856651412
        }
      ]
    ],
    "Counters": [
      1
    ],
    "Guid": 11872291593386833696
  }
]
;--- example.ll
declare void @bar()

define void @an_entrypoint(i32 %a) !guid !0 {
  %t = icmp eq i32 %a, 0
  br i1 %t, label %yes, label %no

yes:
  call void @bar()
  ret void
no:
  ret void
}

attributes #0 = { noinline }
!0 = !{ i64 1000 }