; RUN: opt -S -globalopt < %s | FileCheck %s
; rdar://11022897

; Globalopt should be able to evaluate an invoke.
; CHECK: @tmp = local_unnamed_addr global i32 1

@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]
@tmp = global i32 0

define i32 @one() {
  ret i32 1
}

define void @_GLOBAL__I_a() personality i8* undef {
bb:
  %tmp1 = invoke i32 @one()
          to label %bb2 unwind label %bb4

bb2:                                              ; preds = %bb
  store i32 %tmp1, i32* @tmp
  ret void

bb4:                                              ; preds = %bb
  %tmp5 = landingpad { i8*, i32 }
          filter [0 x i8*] zeroinitializer
  unreachable
}
