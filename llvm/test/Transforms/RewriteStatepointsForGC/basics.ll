; This is a collection of really basic tests for gc.statepoint rewriting.
; RUN:  opt %s -rewrite-statepoints-for-gc -spp-rematerialization-threshold=0 -S | FileCheck %s

declare void @foo()

; Trivial relocation over a single call
define i8 addrspace(1)* @test1(i8 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test1
; CHECK-LABEL: entry:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
entry:
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %obj
}

; Two safepoints in a row (i.e. consistent liveness)
define i8 addrspace(1)* @test2(i8 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: @test2
; CHECK-LABEL: entry:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated1 = call coldcc i8 addrspace(1)*
entry:
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %obj
}

; A simple derived pointer
define i8 @test3(i8 addrspace(1)* %obj) gc "statepoint-example" {
; CHECK-LABEL: entry:
; CHECK-NEXT: getelementptr
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %derived.relocated = call coldcc i8 addrspace(1)*
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
; CHECK-NEXT: load i8, i8 addrspace(1)* %derived.relocated
; CHECK-NEXT: load i8, i8 addrspace(1)* %obj.relocated
entry:
  %derived = getelementptr i8, i8 addrspace(1)* %obj, i64 10
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)

  %a = load i8, i8 addrspace(1)* %derived
  %b = load i8, i8 addrspace(1)* %obj
  %c = sub i8 %a, %b
  ret i8 %c
}

; Tests to make sure we visit both the taken and untaken predeccessor 
; of merge.  This was a bug in the dataflow liveness at one point.
define i8 addrspace(1)* @test4(i1 %cmp, i8 addrspace(1)* %obj) gc "statepoint-example" {
entry:
  br i1 %cmp, label %taken, label %untaken

taken:
; CHECK-LABEL: taken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated = call coldcc i8 addrspace(1)*
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %merge

untaken:
; CHECK-LABEL: untaken:
; CHECK-NEXT: gc.statepoint
; CHECK-NEXT: %obj.relocated1 = call coldcc i8 addrspace(1)*
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  br label %merge

merge:
; CHECK-LABEL: merge:
; CHECK-NEXT: %.0 = phi i8 addrspace(1)* [ %obj.relocated, %taken ], [ %obj.relocated1, %untaken ]
; CHECK-NEXT: ret i8 addrspace(1)* %.0
  ret i8 addrspace(1)* %obj
}

; When run over a function which doesn't opt in, should do nothing!
define i8 addrspace(1)* @test5(i8 addrspace(1)* %obj) gc "ocaml" {
; CHECK-LABEL: @test5
; CHECK-LABEL: entry:
; CHECK-NEXT: gc.statepoint
; CHECK-NOT: %obj.relocated = call coldcc i8 addrspace(1)*
entry:
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* @foo, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0)
  ret i8 addrspace(1)* %obj
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)