; RUN: opt < %s -S -passes=rewrite-statepoints-for-gc | FileCheck %s

declare i64 addrspace(1)* @some_call(i64 addrspace(1)*)
declare i32 @personality_function()

define i64 addrspace(1)* @test_basic(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1) gc "statepoint-example" personality i32 ()* @personality_function {
; CHECK-LABEL: entry:
entry:
  ; CHECK: invoke
  ; CHECK: statepoint
  ; CHECK: some_call
  %ret_val = invoke i64 addrspace(1)* @some_call(i64 addrspace(1)* %obj)
               to label %normal_return unwind label %exceptional_return

; CHECK-LABEL: normal_return:
; CHECK: gc.result
; CHECK: ret i64

normal_return:
  ret i64 addrspace(1)* %ret_val

; CHECK-LABEL: exceptional_return:
; CHECK: landingpad
; CHECK: ret i64

exceptional_return:
  %landing_pad4 = landingpad token
          cleanup
  ret i64 addrspace(1)* %obj1
}

declare <4 x i64 addrspace(1)*> @some_vector_call(<4 x i64 addrspace(1)*>)

define <4 x i64 addrspace(1)*> @test_basic_vector(<4 x i64 addrspace(1)*> %objs, <4 x i64 addrspace(1)*> %objs1) gc "statepoint-example" personality i32 ()* @personality_function {
; CHECK-LABEL: @test_basic_vector
entry:
; CHECK: invoke{{.*}}llvm.experimental.gc.statepoint{{.*}}some_vector_call
  %ret_val = invoke <4 x i64 addrspace(1)*> @some_vector_call(<4 x i64 addrspace(1)*> %objs)
               to label %normal_return unwind label %exceptional_return

; CHECK-LABEL: normal_return:
; CHECK: gc.result
; CHECK: ret <4 x i64 addrspace(1)*>

normal_return:
  ret <4 x i64 addrspace(1)*> %ret_val

; CHECK-LABEL: exceptional_return:
; CHECK: landingpad
; CHECK: ret <4 x i64 addrspace(1)*>

exceptional_return:
  %landing_pad4 = landingpad token
          cleanup
  ret <4 x i64 addrspace(1)*> %objs1
}

define i64 addrspace(1)* @test_two_invokes(i64 addrspace(1)* %obj, i64 addrspace(1)* %obj1) gc "statepoint-example" personality i32 ()* @personality_function {
; CHECK-LABEL: entry:
entry:
  ; CHECK: invoke
  ; CHECK: statepoint
  ; CHECK: some_call
  %ret_val1 = invoke i64 addrspace(1)* @some_call(i64 addrspace(1)* %obj)
               to label %second_invoke unwind label %exceptional_return

; CHECK-LABEL: second_invoke:
second_invoke:
  ; CHECK: invoke
  ; CHECK: statepoint
  ; CHECK: some_call
  %ret_val2 = invoke i64 addrspace(1)* @some_call(i64 addrspace(1)* %ret_val1)
                to label %normal_return unwind label %exceptional_return

; CHECK-LABEL: normal_return:
normal_return:
  ; CHECK: gc.result
  ; CHECK: ret i64
  ret i64 addrspace(1)* %ret_val2

; CHECK: exceptional_return:
; CHECK: ret i64

exceptional_return:
  %landing_pad4 = landingpad token
          cleanup
  ret i64 addrspace(1)* %obj1
}

define i64 addrspace(1)* @test_phi_node(i1 %cond, i64 addrspace(1)* %obj) gc "statepoint-example" personality i32 ()* @personality_function {
; CHECK-LABEL: @test_phi_node
; CHECK-LABEL: entry:
entry:
  br i1 %cond, label %left, label %right

left:
  %ret_val_left = invoke i64 addrspace(1)* @some_call(i64 addrspace(1)* %obj)
                    to label %merge unwind label %exceptional_return

right:
  %ret_val_right = invoke i64 addrspace(1)* @some_call(i64 addrspace(1)* %obj)
                     to label %merge unwind label %exceptional_return

; CHECK: merge[[A:[0-9]]]:
; CHECK: gc.result
; CHECK: br label %[[with_phi:merge[0-9]*]]

; CHECK: merge[[B:[0-9]]]:
; CHECK: gc.result
; CHECK: br label %[[with_phi]]

; CHECK: [[with_phi]]:
; CHECK: phi
; CHECK: ret i64 addrspace(1)* %ret_val
merge:
  %ret_val = phi i64 addrspace(1)* [%ret_val_left, %left], [%ret_val_right, %right]
  ret i64 addrspace(1)* %ret_val

; CHECK-LABEL: exceptional_return:
; CHECK: ret i64 addrspace(1)*

exceptional_return:
  %landing_pad4 = landingpad token
          cleanup
  ret i64 addrspace(1)* %obj
}

declare void @do_safepoint()
define void @gc.safepoint_poll() {
; CHECK-LABEL: gc.safepoint_poll
; CHECK-LABEL: entry
; CHECK-NEXT: do_safepoint
; CHECK-NEXT: ret void
entry:
  call void @do_safepoint()
  ret void
}
