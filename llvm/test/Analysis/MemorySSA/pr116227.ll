; RUN: opt -disable-output -passes="loop-mssa(simple-loop-unswitch<nontrivial>),print<memoryssa>" -verify-memoryssa < %s 2>&1 | FileCheck %s

declare ptr @malloc() allockind("alloc,uninitialized")

; CHECK-LABEL: MemorySSA for function: test

; CHECK: for.body.us:
; CHECK-NEXT: 3 = MemoryPhi({entry.split.us,liveOnEntry},{for.body.us,3})

; CHECK: for.body:
; CHECK-NEXT: 2 = MemoryPhi({entry.split,liveOnEntry},{for.body,1})
; CHECK-NEXT: 1 = MemoryDef(2)
; CHECK-NEXT:  %call.i = call ptr @malloc()

define void @test(i1 %arg) {
entry:
  br label %for.body

for.body:
  %call.i = call ptr @malloc()
  %cmp.i = icmp ne ptr %call.i, null
  %or.cond.i = select i1 %cmp.i, i1 %arg, i1 false
  br i1 %or.cond.i, label %exit, label %for.body

exit:
  ret void
}

; CHECK-LABEL: MemorySSA for function: test_extra_defs

; CHECK: entry:
; CHECK-NEXT: 1 = MemoryDef(liveOnEntry)
; CHECK-NEXT: store i8 1, ptr %p, align 1

; CHECK: for.body.us:
; CHECK-NEXT: 5 = MemoryPhi({entry.split.us,1},{for.body.us,6})
; CHECK-NEXT: 6 = MemoryDef(5)
; CHECK-NEXT: store i8 2, ptr %p, align 1

; CHECK: for.body:
; CHECK-NEXT: 4 = MemoryPhi({entry.split,1},{for.body,3})
; CHECK-NEXT: 2 = MemoryDef(4)
; CHECK-NEXT: store i8 2, ptr %p
; CHECK-NEXT: 3 = MemoryDef(2)
; CHECK-NEXT: %call.i = call ptr @malloc()

define void @test_extra_defs(ptr %p, i1 %arg) {
entry:
  store i8 1, ptr %p
  br label %for.body

for.body:
  store i8 2, ptr %p
  %call.i = call ptr @malloc()
  %cmp.i = icmp ne ptr %call.i, null
  %or.cond.i = select i1 %cmp.i, i1 %arg, i1 false
  br i1 %or.cond.i, label %exit, label %for.body

exit:
  ret void
}
