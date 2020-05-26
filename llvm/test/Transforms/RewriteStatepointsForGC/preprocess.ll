; RUN: opt -rewrite-statepoints-for-gc -S < %s | FileCheck %s
; RUN: opt -passes=rewrite-statepoints-for-gc -S < %s | FileCheck %s

; Test to make sure we destroy LCSSA's single entry phi nodes before
; running liveness

declare void @consume(...) "gc-leaf-function"

define void @test6(i64 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test6
entry:
  br label %next

next:                                             ; preds = %entry
; CHECK-LABEL: next:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: gc.relocate
; CHECK-NEXT: bitcast
; CHECK-NEXT: @consume(i64 addrspace(1)* %obj.relocated.casted)
; CHECK-NEXT: @consume(i64 addrspace(1)* %obj.relocated.casted)
; Need to delete unreachable gc.statepoint call
  %obj2 = phi i64 addrspace(1)* [ %obj, %entry ]
  call void @foo() [ "deopt"() ]
  call void (...) @consume(i64 addrspace(1)* %obj2)
  call void (...) @consume(i64 addrspace(1)* %obj)
  ret void
}

define void @test7() gc "statepoint-example" {
; CHECK-LABEL: test7
; CHECK-NOT: gc.statepoint
; Need to delete unreachable gc.statepoint invoke - tested separately given
; a correct implementation could only remove the instructions, not the block
  ret void

unreached:                                        ; preds = %unreached
  %obj = phi i64 addrspace(1)* [ null, %unreached ]
  call void @foo() [ "deopt"() ]
  call void (...) @consume(i64 addrspace(1)* %obj)
  br label %unreached
}

define void @test8() gc "statepoint-example" personality i32 ()* undef {
; CHECK-LABEL: test8
; CHECK-NOT: gc.statepoint
; Bound the last check-not
  ret void

unreached:                                        ; No predecessors!
  invoke void @foo() [ "deopt"() ]
; CHECK-LABEL: @foo
          to label %normal_return unwind label %exceptional_return

normal_return:                                    ; preds = %unreached
  ret void

exceptional_return:                               ; preds = %unreached
  %landing_pad4 = landingpad { i8*, i32 }
          cleanup
  ret void
}

declare void @foo()
