; RUN: opt -rewrite-statepoints-for-gc -S < %s | FileCheck %s

declare void @consume(...)

; Test to make sure we destroy LCSSA's single entry phi nodes before
; running liveness
define void @test6(i64 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test6
entry:
  br label %next

next:
; CHECK-LABEL: next:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: @consume(i64 addrspace(1)* %obj.relocated.casted)
; CHECK-NEXT: @consume(i64 addrspace(1)* %obj.relocated.casted)
  %obj2 = phi i64 addrspace(1)* [ %obj, %entry ]
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  call void (...) @consume(i64 addrspace(1)* %obj2)
  call void (...) @consume(i64 addrspace(1)* %obj)
  ret void
}

declare void @some_call(i64 addrspace(1)*)

; Need to delete unreachable gc.statepoint call
define void @test7() gc "statepoint-example" {
; CHECK-LABEL: test7
; CHECK-NOT: gc.statepoint
  ret void

unreached:
  %obj = phi i64 addrspace(1)* [null, %unreached]
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
  call void (...) @consume(i64 addrspace(1)* %obj)
  br label %unreached
}

; Need to delete unreachable gc.statepoint invoke - tested seperately given
; a correct implementation could only remove the instructions, not the block
define void @test8() gc "statepoint-example" personality i32 ()* undef {
; CHECK-LABEL: test8
; CHECK-NOT: gc.statepoint
  ret void

unreached:
  invoke token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 0)
          to label %normal_return unwind label %exceptional_return

normal_return:                                    ; preds = %entry
  ret void

exceptional_return:                               ; preds = %entry
  %landing_pad4 = landingpad { i8*, i32 }
          cleanup
  ret void
}

declare void @foo()
; Bound the last check-not
; CHECK-LABEL: @foo

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
