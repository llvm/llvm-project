; REQUIRES: x86_64-linux
;
; RUN: rm -rf %t
; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromJSON --input=%t/profile.json --output=%t/profile.ctxprofdata
; RUN: not opt -passes='require<ctx-prof-analysis>,print<ctx-prof-analysis>' \
; RUN:   %t/example.ll -S 2>&1 | FileCheck %s --check-prefix=NO-FILE

; RUN: not opt -passes='require<ctx-prof-analysis>,print<ctx-prof-analysis>' \
; RUN:   -use-ctx-profile=does_not_exist.ctxprofdata %t/example.ll -S 2>&1 | FileCheck %s --check-prefix=NO-FILE

; RUN: opt -module-summary -passes='thinlto-pre-link<O2>' \
; RUN:   -use-ctx-profile=%t/profile.ctxprofdata %t/example.ll -S -o %t/prelink.ll

; RUN: opt -module-summary -passes='thinlto-pre-link<O2>' -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:  %t/example.ll -S -o %t/prelink.ll
; RUN: opt -passes='require<ctx-prof-analysis>,print<ctx-prof-analysis>' \
; RUN:   -use-ctx-profile=%t/profile.ctxprofdata %t/prelink.ll -S 2> %t/output.txt
; RUN: diff %t/expected-profile-output.txt %t/output.txt

; NO-FILE: error: could not open contextual profile file
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
;--- expected-profile-output.txt
Function Info:
4909520559318251808 : an_entrypoint. MaxCounterID: 2. MaxCallsiteID: 1
12074870348631550642 : another_entrypoint_no_callees. MaxCounterID: 1. MaxCallsiteID: 0
11872291593386833696 : foo. MaxCounterID: 1. MaxCallsiteID: 1

Current Profile:
[
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
  },
  {
    "Counters": [
      5
    ],
    "Guid": 12074870348631550642
  }
]

Flat Profile:
728453322856651412 : 6 7 
12074870348631550642 : 5 
11872291593386833696 : 1 
;--- example.ll
declare void @bar()

define private void @foo(i32 %a, ptr %fct) #0 !guid !0 {
  %t = icmp eq i32 %a, 0
  br i1 %t, label %yes, label %no
yes:
  call void %fct(i32 %a)
  br label %exit
no:
  call void @bar()
  br label %exit
exit:
  ret void
}

define void @an_entrypoint(i32 %a) {
  %t = icmp eq i32 %a, 0
  br i1 %t, label %yes, label %no

yes:
  call void @foo(i32 1, ptr null)
  ret void
no:
  ret void
}

define void @another_entrypoint_no_callees(i32 %a) {
  %t = icmp eq i32 %a, 0
  br i1 %t, label %yes, label %no

yes:
  ret void
no:
  ret void
}

attributes #0 = { noinline }
!0 = !{ i64 11872291593386833696 }