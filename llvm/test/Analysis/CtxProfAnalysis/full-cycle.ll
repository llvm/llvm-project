; REQUIRES: x86_64-linux
;
; RUN: rm -rf %t
; RUN: split-file %s %t
;
; Test that the GUID metadata survives through thinlink.
;
; RUN: llvm-ctxprof-util fromJSON --input=%t/profile.json --output=%t/profile.ctxprofdata
;
; RUN: opt -module-summary -passes='thinlto-pre-link<O2>' -use-ctx-profile=%t/profile.ctxprofdata -o %t/m1.bc %t/m1.ll
; RUN: opt -module-summary -passes='thinlto-pre-link<O2>' -use-ctx-profile=%t/profile.ctxprofdata -o %t/m2.bc %t/m2.ll
;
; RUN: rm -rf %t/postlink
; RUN: mkdir %t/postlink
;
;
; RUN: llvm-lto2 run %t/m1.bc %t/m2.bc -o %t/ -thinlto-distributed-indexes \
; RUN:  -use-ctx-profile=%t/profile.ctxprofdata \
; RUN:  -r %t/m1.bc,f1,plx \
; RUN:  -r %t/m2.bc,f1 \
; RUN:  -r %t/m2.bc,entrypoint,plx
; RUN: opt --passes='function-import,require<ctx-prof-analysis>,print<ctx-prof-analysis>' \
; RUN:  -summary-file=%t/m2.bc.thinlto.bc -use-ctx-profile=%t/profile.ctxprofdata %t/m2.bc \
; RUN:  -S -o %t/m2.post.ll 2> %t/profile.txt
; RUN: diff %t/expected.txt %t/profile.txt
;--- m1.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

source_filename = "random_path/m1.cc"

define private void @f2() #0 !guid !0 {
  ret void
}

define void @f1() #0 {
  call void @f2()
  ret void
}

attributes #0 = { noinline }
!0 = !{ i64 3087265239403591524 }

;--- m2.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

source_filename = "random_path/m2.cc"

declare void @f1()

define void @entrypoint() {
  call void @f1()
  ret void
}
;--- profile.json
[
  {
    "Callsites": [
      [
        {
          "Callsites": [
            [
              {
                "Counters": [
                  10
                ],
                "Guid": 3087265239403591524
              }
            ]
          ],
          "Counters": [
            7
          ],
          "Guid": 2072045998141807037
        }
      ]
    ],
    "Counters": [
      1
    ],
    "Guid": 10507721908651011566
  }
]
;--- expected.txt
Function Info:
10507721908651011566 : entrypoint. MaxCounterID: 1. MaxCallsiteID: 1
3087265239403591524 : f2.llvm.0. MaxCounterID: 1. MaxCallsiteID: 0
2072045998141807037 : f1. MaxCounterID: 1. MaxCallsiteID: 1

Current Profile:
[
  {
    "Callsites": [
      [
        {
          "Callsites": [
            [
              {
                "Counters": [
                  10
                ],
                "Guid": 3087265239403591524
              }
            ]
          ],
          "Counters": [
            7
          ],
          "Guid": 2072045998141807037
        }
      ]
    ],
    "Counters": [
      1
    ],
    "Guid": 10507721908651011566
  }
]
