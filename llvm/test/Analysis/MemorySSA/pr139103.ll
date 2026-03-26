; RUN: opt -disable-output -passes="loop-mssa(licm,loop-rotate,licm,simple-loop-unswitch<nontrivial>),print<memoryssa>" -verify-memoryssa < %s 2>&1 | FileCheck %s

; Make sure that we update MSSA correctly in this case.

; CHECK-LABEL: MemorySSA for function: test
; CHECK: for.header2.preheader:
; CHECK-NEXT: 11 = MemoryPhi({entry.split,liveOnEntry},{for.header,9})
; CHECK: for.body.us:
; CHECK-NEXT: 7 = MemoryPhi({for.header2.preheader.split.us,11},{for.header2.us,9})
; CHECK-NEXT: 8 = MemoryDef(7)->7
; CHECK-NEXT: store i32 0, ptr %p, align 4
; CHECK-NEXT: 9 = MemoryDef(8)->8
; CHECK-NEXT: store i8 0, ptr %p, align 1

define void @test(ptr %p, i1 %cond) {
entry:
  br label %for.header

for.header:
  br i1 false, label %exit.loopexit1, label %for.header2.preheader

for.header2.preheader:
  br label %for.body

for.header2:
  br i1 false, label %for.latch, label %for.body

for.body:
  store i32 0, ptr %p, align 4
  store i8 0, ptr %p, align 1
  br i1 %cond, label %for.header2, label %exit.loopexit

for.latch:
  br i1 false, label %for.inc, label %exit.loopexit1

for.inc:
  br label %for.header

exit.loopexit:
  br label %exit

exit.loopexit1:
  br label %exit

exit:
  ret void
}
