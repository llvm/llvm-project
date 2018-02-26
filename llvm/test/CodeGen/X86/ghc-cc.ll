; RUN: llc < %s -tailcallopt -mtriple=i686-linux-gnu | FileCheck %s

; Test the GHC call convention works (x86-32)

@base = external global i32 ; assigned to register: ebx
@sp   = external global i32 ; assigned to register: ebp
@hp   = external global i32 ; assigned to register: edi
@r1   = external global i32 ; assigned to register: esi

define void @zap(i32 %a, i32 %b) nounwind {
entry:
  ; CHECK: movl {{[0-9]*}}(%esp), %ebx
  ; CHECK-NEXT: movl {{[0-9]*}}(%esp), %ebp
  ; CHECK-NEXT: calll addtwo
  %0 = call ghccc i32 @addtwo(i32 %a, i32 %b)
  ; CHECK: calll foo
  call void @foo() nounwind
  ret void
}

define ghccc i32 @addtwo(i32 %x, i32 %y) nounwind {
entry:
  ; CHECK: leal (%ebx,%ebp), %eax
  %0 = add i32 %x, %y
  ; CHECK-NEXT: ret
  ret i32 %0
}

define ghccc void @foo() nounwind {
entry:
  ; CHECK:      movl r1, %esi
  ; CHECK-NEXT: movl hp, %edi
  ; CHECK-NEXT: movl sp, %ebp
  ; CHECK-NEXT: movl base, %ebx
  %0 = load i32, i32* @r1
  %1 = load i32, i32* @hp
  %2 = load i32, i32* @sp
  %3 = load i32, i32* @base
  ; CHECK: jmp bar
  tail call ghccc void @bar( i32 %3, i32 %2, i32 %1, i32 %0 ) nounwind
  ret void
}

declare ghccc void @bar(i32, i32, i32, i32)
