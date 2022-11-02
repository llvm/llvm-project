; RUN: opt -passes='loop(simple-loop-unswitch<nontrivial>),verify<loops>' -S < %s | FileCheck %s
; RUN: opt -passes='loop-mssa(simple-loop-unswitch<nontrivial>),verify<loops>' -S < %s | FileCheck %s
; RUN: opt -simple-loop-unswitch -enable-nontrivial-unswitch -verify-memoryssa -S < %s | FileCheck %s

declare ptr @pluto()
declare void @llvm.experimental.guard(i1, ...)
declare void @widget()

; REQUIRES: asserts
; XFAIL: *

define void @foo(ptr addrspace(1) %arg, i64 %arg1) personality ptr @pluto {
bb:
  %tmp = icmp slt i32 poison, 570
  %tmp2 = select i1 %tmp, i1 true, i1 false
  br label %bb3

bb3:                                              ; preds = %bb6, %bb
  call void (i1, ...) @llvm.experimental.guard(i1 %tmp2, i32 7) [ "deopt"() ]
  invoke void @widget()
          to label %bb4 unwind label %bb7

bb4:                                              ; preds = %bb3
  invoke void @widget()
          to label %bb6 unwind label %bb7

bb6:                                              ; preds = %bb4
  invoke void @widget()
          to label %bb3 unwind label %bb7

bb7:                                              ; preds = %bb6, %bb4, %bb3
  %tmp8 = landingpad { ptr, i32 }
          cleanup
  ret void
}

