; RUN: opt -rewrite-statepoints-for-gc -verify -S < %s | FileCheck %s
; RUN: opt -passes=rewrite-statepoints-for-gc,verify -S < %s | FileCheck %s

declare i8 addrspace(1)* @gc_call()

declare i32* @fake_personality_function()

define i8 addrspace(1)* @test(i1 %c) gc "statepoint-example" personality i32* ()* @fake_personality_function {
; CHECK-LABEL: @test(
entry:
  br i1 %c, label %gc_invoke, label %normal_dest

gc_invoke:
; CHECK: [[TOKEN:%[^ ]+]] = invoke token {{[^@]+}}@llvm.experimental.gc.statepoint{{[^@]+}}@gc_call
  %obj = invoke i8 addrspace(1)* @gc_call() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
          to label %normal_dest unwind label %unwind_dest

unwind_dest:
; CHECK: unwind_dest:
  %lpad = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef

; CHECK: [[NORMAL_DEST_SPLIT:[^:]+:]]
; CHECK-NEXT: [[RET_VAL:%[^ ]+]] = call i8 addrspace(1)* @llvm.experimental.gc.result.p1i8(token [[TOKEN]])
; CHECK-NEXT: br label %normal_dest

normal_dest:
; CHECK: normal_dest:
; CHECK-NEXT: %merge = phi i8 addrspace(1)* [ null, %entry ], [ %obj2, %normal_dest1 ]
  %merge = phi i8 addrspace(1)* [ null, %entry ], [ %obj, %gc_invoke ]
  ret i8 addrspace(1)* %merge
}
