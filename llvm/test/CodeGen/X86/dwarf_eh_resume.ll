; RUN: opt -mtriple=x86_64-linux-gnu -dwarfehprepare -S %s | FileCheck %s

declare i32 @hoge(...)

; Check that 'resume' is lowered to _Unwind_Resume which marked as 'noreturn'
define void @pluto() align 2 personality ptr @hoge {
;CHECK: call void @_Unwind_Resume(ptr %exn.obj) [[A:#.*]]
;CHECK: attributes [[A]] = { noreturn }
bb:
  invoke void @spam()
          to label %bb1 unwind label %bb2

bb1:                                              ; preds = %bb
  ret void

bb2:                                              ; preds = %bb
  %tmp = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %tmp

}

declare void @spam()
