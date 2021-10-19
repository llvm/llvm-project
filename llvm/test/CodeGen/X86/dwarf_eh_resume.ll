; RUN: opt -mtriple=x86_64-linux-gnu -dwarfehprepare -S %s | FileCheck %s

declare i32 @hoge(...)

; Check that 'resume' is lowered to _Unwind_Resume which marked as 'noreturn'
define void @pluto() align 2 personality i8* bitcast (i32 (...)* @hoge to i8*) {
;CHECK: call void @_Unwind_Resume(i8* %exn.obj) [[A:#.*]]
;CHECK: attributes [[A]] = { noreturn }
bb:
  invoke void @spam()
          to label %bb1 unwind label %bb2

bb1:                                              ; preds = %bb
  ret void

bb2:                                              ; preds = %bb
  %tmp = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } %tmp

}

declare void @spam()
