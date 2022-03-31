; This is a collection of really basic tests for gc.statepoint rewriting.
; RUN: opt < %s -rewrite-statepoints-for-gc -spp-rematerialization-threshold=0 -S | FileCheck %s
; RUN: opt < %s -passes=rewrite-statepoints-for-gc -spp-rematerialization-threshold=0 -S | FileCheck %s

; Trivial relocation over a single call

declare void @foo()

define i8 addrspace(1)* @test1(i8 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test1
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
; Two safepoints in a row (i.e. consistent liveness)
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %obj
}

define i8 addrspace(1)* @test2(i8 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test2
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated2 = call coldcc i8 addrspace(1)*
; A simple derived pointer
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  ret i8 addrspace(1)* %obj
}

define i8 @test3(i8 addrspace(1)* %obj) gc "statepoint-example" {
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT: getelementptr
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
; CHECK-NEXT: %derived.relocated = call coldcc i8 addrspace(1)*
; CHECK-NEXT: load i8, i8 addrspace(1)* %derived.relocated
; CHECK-NEXT: load i8, i8 addrspace(1)* %obj.relocated
; Tests to make sure we visit both the taken and untaken predeccessor 
; of merge.  This was a bug in the dataflow liveness at one point.
  %derived = getelementptr i8, i8 addrspace(1)* %obj, i64 10
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  %a = load i8, i8 addrspace(1)* %derived
  %b = load i8, i8 addrspace(1)* %obj
  %c = sub i8 %a, %b
  ret i8 %c
}

define i8 addrspace(1)* @test4(i1 %cmp, i8 addrspace(1)* %obj) gc "statepoint-example" {
entry:
  br i1 %cmp, label %taken, label %untaken

taken:                                            ; preds = %entry
; CHECK-LABEL: taken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  br label %merge

untaken:                                          ; preds = %entry
; CHECK-LABEL: untaken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated2 = call coldcc i8 addrspace(1)*
  call void @foo() [ "deopt"(i32 0, i32 -1, i32 0, i32 0, i32 0) ]
  br label %merge

merge:                                            ; preds = %untaken, %taken
; CHECK-LABEL: merge:
; CHECK-NEXT: %.0 = phi i8 addrspace(1)* [ %obj.relocated, %taken ], [ %obj.relocated2, %untaken ]
; CHECK-NEXT: ret i8 addrspace(1)* %.0
; When run over a function which doesn't opt in, should do nothing!
  ret i8 addrspace(1)* %obj
}

define i8 addrspace(1)* @test5(i8 addrspace(1)* %obj) gc "ocaml" {
; CHECK-LABEL: @test5
entry:
; CHECK-LABEL: entry:
; CHECK-NEXT: gc.statepoint
; CHECK-NOT: %obj.relocated = call coldcc i8 addrspace(1)*
  %0 = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* elementtype(void ()) @foo, i32 0, i32 0, i32 0, i32 0) ["deopt" (i32 0, i32 -1, i32 0, i32 0, i32 0)]
  ret i8 addrspace(1)* %obj
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)
