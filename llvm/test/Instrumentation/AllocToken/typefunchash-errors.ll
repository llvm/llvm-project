; Test that metadata without function name (generated for another mode) is
; rejected in the typefunchash modes.
; RUN: not opt < %s -passes='inferattrs,alloc-token<mode=typefunchash>' -disable-output 2>&1 | FileCheck %s --check-prefix=NOFUNC
; RUN: not opt < %s -passes='inferattrs,alloc-token<mode=typefunchashpointersplit>' -disable-output 2>&1 | FileCheck %s --check-prefix=NOFUNC-SPLIT

; NOFUNC: error: !alloc_token without function name is incompatible with mode typefunchash{{$}}
; NOFUNC-SPLIT: error: !alloc_token without function name is incompatible with mode typefunchashpointersplit{{$}}

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

declare ptr @malloc(i64)

define ptr @test_no_function_name() sanitize_alloc_token {
entry:
  %ptr = call ptr @malloc(i64 4), !alloc_token !0
  ret ptr %ptr
}

!0 = !{!"int", i1 false}
