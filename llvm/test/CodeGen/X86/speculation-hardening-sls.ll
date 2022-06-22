; RUN: llc -mattr=harden-sls-ret -mtriple=x86_64-unknown-unknown < %s | FileCheck %s -check-prefixes=CHECK,RET
; RUN: llc -mattr=harden-sls-ijmp -mtriple=x86_64-unknown-unknown < %s | FileCheck %s -check-prefixes=CHECK,IJMP

define dso_local i32 @double_return(i32 %a, i32 %b) local_unnamed_addr {
; CHECK-LABEL: double_return:
; CHECK:         jle
; CHECK-NOT:     int3
; CHECK:         retq
; RET-NEXT:      int3
; IJMP-NOT:      int3
; CHECK:         retq
; RET-NEXT:      int3
; IJMP-NOT:      int3
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %div = sdiv i32 %a, %b
  ret i32 %div

if.else:                                          ; preds = %entry
  %div1 = sdiv i32 %b, %a
  ret i32 %div1
}

@__const.indirect_branch.ptr = private unnamed_addr constant [2 x ptr] [ptr blockaddress(@indirect_branch, %return), ptr blockaddress(@indirect_branch, %l2)], align 8

; Function Attrs: norecurse nounwind readnone
define dso_local i32 @indirect_branch(i32 %a, i32 %b, i32 %i) {
; CHECK-LABEL: indirect_branch:
; CHECK:         jmpq *
; RET-NOT:       int3
; IJMP-NEXT:     int3
; CHECK:         retq
; RET-NEXT:      int3
; IJMP-NOT:      int3
; CHECK:         retq
; RET-NEXT:      int3
; IJMP-NOT:      int3
entry:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds [2 x ptr], ptr @__const.indirect_branch.ptr, i64 0, i64 %idxprom
  %0 = load ptr, ptr %arrayidx, align 8
  indirectbr ptr %0, [label %return, label %l2]

l2:                                               ; preds = %entry
  br label %return

return:                                           ; preds = %entry, %l2
  %retval.0 = phi i32 [ 1, %l2 ], [ 0, %entry ]
  ret i32 %retval.0
}

define i32 @asmgoto() {
; CHECK-LABEL: asmgoto:
; CHECK:       # %bb.0: # %entry
; CHECK:         jmp .L
; CHECK-NOT:     int3
; CHECK:         retq
; RET-NEXT:      int3
; IJMP-NOT:      int3
; CHECK:         retq
; RET-NEXT:      int3
; IJMP-NOT:      int3
entry:
  callbr void asm sideeffect "jmp $0", "X"(ptr blockaddress(@asmgoto, %d))
            to label %asm.fallthrough [label %d]
     ; The asm goto above produces a direct branch:

asm.fallthrough:               ; preds = %entry
  ret i32 0

d:                             ; preds = %asm.fallthrough, %entry
  ret i32 1
}

define void @bar(ptr %0) {
; CHECK-LABEL: bar:
; CHECK:         jmpq *
; RET-NOT:       int3
; IJMP-NEXT:     int3
; CHECK-NOT:     ret
  tail call void %0()
  ret void
}

declare dso_local void @foo()

define dso_local void @bar2() {
; CHECK-LABEL: bar2:
; CHECK:         jmp foo
; CHECK-NOT:     int3
; CHECK-NOT:     ret
  tail call void @foo()
  ret void
}
