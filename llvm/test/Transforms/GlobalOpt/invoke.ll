; RUN: opt -S -passes=globalopt < %s | FileCheck %s
; rdar://11022897

; Globalopt should be able to evaluate an invoke.
; CHECK: @tmp = local_unnamed_addr global i32 1

@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__I_a, ptr null }]
@tmp = global i32 0

define i32 @one() {
  ret i32 1
}

define void @_GLOBAL__I_a() personality ptr undef {
bb:
  %tmp1 = invoke i32 @one()
          to label %bb2 unwind label %bb4

bb2:                                              ; preds = %bb
  store i32 %tmp1, ptr @tmp
  ret void

bb4:                                              ; preds = %bb
  %tmp5 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  unreachable
}
