; REQUIRES: x86_64-linux

; RUN: split-file %s %t
; RUN: llvm-ctxprof-util fromJSON --input=%t/profile.json --output=%t/profile.ctxprofdata
; RUN: not opt -passes='require<ctx-prof-analysis>,print<ctx-prof-analysis>' \
; RUN:   %t/empty.ll -S 2>&1 | FileCheck %s --check-prefix=NO-FILE

; RUN: not opt -passes='require<ctx-prof-analysis>,print<ctx-prof-analysis>' \
; RUN:   -use-ctx-profile=does_not_exist.ctxprofdata %t/empty.ll -S 2>&1 | FileCheck %s --check-prefix=NO-FILE

; RUN: opt -passes='require<ctx-prof-analysis>,print<ctx-prof-analysis>' \
; RUN:   -use-ctx-profile=%t/profile.ctxprofdata %t/empty.ll -S 2> %t/output.json
; RUN: diff %t/profile.json %t/output.json

; NO-FILE: error: could not open contextual profile file
;
; This is the reference profile, laid out in the format the json formatter will
; output it from opt.
;--- profile.json
[
  {
    "Callsites": [
      [],
      [
        {
          "Counters": [
            4,
            5
          ],
          "Guid": 2000
        },
        {
          "Counters": [
            6,
            7,
            8
          ],
          "Guid": 18446744073709551613
        }
      ]
    ],
    "Counters": [
      1,
      2,
      3
    ],
    "Guid": 1000
  },
  {
    "Counters": [
      5,
      9,
      10
    ],
    "Guid": 18446744073709551612
  }
]
;--- empty.ll
