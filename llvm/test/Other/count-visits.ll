; RUN: opt -passes=count-visits -stats 2>&1 -disable-output < %s | FileCheck %s --check-prefix=ONE
; RUN: opt -passes='cgscc(count-visits)' -stats 2>&1 -disable-output < %s | FileCheck %s --check-prefix=ONE
; RUN: opt -passes='cgscc(count-visits,instcombine)' -stats 2>&1 -disable-output < %s | FileCheck %s --check-prefix=TWO
; RUN: opt -passes='default<O1>' -count-cgscc-max-visits -stats 2>&1 -disable-output < %s | FileCheck %s --check-prefix=PIPELINE
; RUN: opt -passes='default<O3>' -count-cgscc-max-visits -stats 2>&1 -disable-output < %s | FileCheck %s --check-prefix=PIPELINE

; ONE: 1 count-visits - Max number of times we visited a function
; TWO: 2 count-visits - Max number of times we visited a function
; PIPELINE: count-visits - Max number of times we visited a function

define void @f() {
  %a = bitcast ptr @g to ptr
  call void %a()
  ret void
}

define void @g() {
  call void @f()
  ret void
}
