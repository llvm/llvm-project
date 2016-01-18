; RUN: llc -verify-machineinstrs -mtriple=i386-linux-gnu %s -o - | FileCheck %s -check-prefix=i386
; RUN: llc -verify-machineinstrs -mtriple=i386-linux-gnu -pre-RA-sched=fast %s -o - | FileCheck %s -check-prefix=i386f
; RUN: llc -verify-machineinstrs -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s -check-prefix=x8664
; RUN: llc -verify-machineinstrs -mtriple=x86_64-linux-gnu -pre-RA-sched=fast %s -o - | FileCheck %s -check-prefix=x8664

declare i32 @foo()
declare i32 @bar(i64)

define i64 @test_intervening_call(i64* %foo, i64 %bar, i64 %baz) {
; i386-LABEL: test_intervening_call:
; i386: cmpxchg8b
; i386-NEXT: pushl %eax
; i386-NEXT: seto %al
; i386-NEXT: lahf
; i386-NEXT: movl %eax, [[FLAGS:%.*]]
; i386-NEXT: popl %eax
; i386-NEXT: movl %edx, 4(%esp)
; i386-NEXT: movl %eax, (%esp)
; i386-NEXT: calll bar
; i386-NEXT: movl [[FLAGS]], %eax
; i386-NEXT: addb $127, %al
; i386-NEXT: sahf
; i386-NEXT: jne

; i386f-LABEL: test_intervening_call:
; i386f: cmpxchg8b
; i386f-NEXT: movl %eax, (%esp)
; i386f-NEXT: movl %edx, 4(%esp)
; i386f-NEXT: seto %al
; i386f-NEXT: lahf
; i386f-NEXT: movl %eax, [[FLAGS:%.*]]
; i386f-NEXT: calll bar
; i386f-NEXT: movl [[FLAGS]], %eax
; i386f-NEXT: addb $127, %al
; i386f-NEXT: sahf
; i386f-NEXT: jne

; x8664-LABEL: test_intervening_call:
; x8664: cmpxchgq
; x8664: pushq %rax
; x8664-NEXT: seto %al
; x8664-NEXT: lahf
; x8664-NEXT: movq %rax, [[FLAGS:%.*]]
; x8664-NEXT: popq %rax
; x8664-NEXT: movq %rax, %rdi
; x8664-NEXT: callq bar
; x8664-NEXT: movq [[FLAGS]], %rax
; x8664-NEXT: addb $127, %al
; x8664-NEXT: sahf
; x8664-NEXT: jne

  %cx = cmpxchg i64* %foo, i64 %bar, i64 %baz seq_cst seq_cst
  %v = extractvalue { i64, i1 } %cx, 0
  %p = extractvalue { i64, i1 } %cx, 1
  call i32 @bar(i64 %v)
  br i1 %p, label %t, label %f

t:
  ret i64 42

f:
  ret i64 0
}

; Interesting in producing a clobber without any function calls.
define i32 @test_control_flow(i32* %p, i32 %i, i32 %j) {
; i386-LABEL: test_control_flow:
; i386: cmpxchg
; i386-NEXT: jne

; i386f-LABEL: test_control_flow:
; i386f: cmpxchg
; i386f-NEXT: jne

; x8664-LABEL: test_control_flow:
; x8664: cmpxchg
; x8664-NEXT: jne

entry:
  %cmp = icmp sgt i32 %i, %j
  br i1 %cmp, label %loop_start, label %cond.end

loop_start:
  br label %while.condthread-pre-split.i

while.condthread-pre-split.i:
  %.pr.i = load i32, i32* %p, align 4
  br label %while.cond.i

while.cond.i:
  %0 = phi i32 [ %.pr.i, %while.condthread-pre-split.i ], [ 0, %while.cond.i ]
  %tobool.i = icmp eq i32 %0, 0
  br i1 %tobool.i, label %while.cond.i, label %while.body.i

while.body.i:
  %.lcssa = phi i32 [ %0, %while.cond.i ]
  %1 = cmpxchg i32* %p, i32 %.lcssa, i32 %.lcssa seq_cst seq_cst
  %2 = extractvalue { i32, i1 } %1, 1
  br i1 %2, label %cond.end.loopexit, label %while.condthread-pre-split.i

cond.end.loopexit:
  br label %cond.end

cond.end:
  %cond = phi i32 [ %i, %entry ], [ 0, %cond.end.loopexit ]
  ret i32 %cond
}

; This one is an interesting case because CMOV doesn't have a chain
; operand. Naive attempts to limit cmpxchg EFLAGS use are likely to fail here.
define i32 @test_feed_cmov(i32* %addr, i32 %desired, i32 %new) {
; i386-LABEL: test_feed_cmov:
; i386: cmpxchgl
; i386-NEXT: seto %al
; i386-NEXT: lahf
; i386-NEXT: movl %eax, [[FLAGS:%.*]]
; i386-NEXT: calll foo
; i386-NEXT: pushl %eax
; i386-NEXT: movl [[FLAGS]], %eax
; i386-NEXT: addb $127, %al
; i386-NEXT: sahf
; i386-NEXT: popl %eax

; i386f-LABEL: test_feed_cmov:
; i386f: cmpxchgl
; i386f-NEXT: seto %al
; i386f-NEXT: lahf
; i386f-NEXT: movl %eax, [[FLAGS:%.*]]
; i386f-NEXT: calll foo
; i386f-NEXT: pushl %eax
; i386f-NEXT: movl [[FLAGS]], %eax
; i386f-NEXT: addb $127, %al
; i386f-NEXT: sahf
; i386f-NEXT: popl %eax

; x8664-LABEL: test_feed_cmov:
; x8664: cmpxchgl
; x8664: seto %al
; x8664-NEXT: lahf
; x8664-NEXT: movq %rax, [[FLAGS:%.*]]
; x8664-NEXT: callq foo
; x8664-NEXT: pushq %rax
; x8664-NEXT: movq [[FLAGS]], %rax
; x8664-NEXT: addb $127, %al
; x8664-NEXT: sahf
; x8664-NEXT: popq %rax

  %res = cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  %success = extractvalue { i32, i1 } %res, 1

  %rhs = call i32 @foo()

  %ret = select i1 %success, i32 %new, i32 %rhs
  ret i32 %ret
}
