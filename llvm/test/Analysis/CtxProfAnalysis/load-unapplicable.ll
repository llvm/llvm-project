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
; output it from opt. Note that the root GUIDs - 12341 and 34234 - are different from
; the GUID present in the module, which is otherwise present in the profile, but not
; as a root.
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
    "Guid": 1000
  },
  {
    "Callsites": [
      [
        {
          "Counters": [
            6,
            7
          ],
          "Guid": 1000
        }
      ]
    ],
    "Counters": [
      1
    ],
    "Guid": 34234
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
