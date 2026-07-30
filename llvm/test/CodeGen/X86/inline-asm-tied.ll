; RUN: llc < %s -mtriple=i386-apple-darwin9 -O0 -optimize-regalloc -regalloc=basic -no-integrated-as | FileCheck %s
; rdar://6992609

target triple = "i386-apple-darwin9.0"

define i64 @_OSSwapInt64(i64 %_data) nounwind {
entry:
  %0 = call i64 asm "bswap   %eax\0A\09bswap   %edx\0A\09xchgl   %eax, %%edx", "=A,0,~{dirflag},~{fpsr},~{flags}"(i64 %_data) nounwind
  ret i64 %0
}

; CHECK-LABEL: __OSSwapInt64:
; CHECK-DAG: movl 8(%esp), %edx
; CHECK-DAG: movl 4(%esp), %eax
; CHECK: ## InlineAsm Start
; CHECK: ## InlineAsm End
;       Everything is set up in eax:edx, return immediately.
; CHECK-NEXT: retl

; The tied operands are not necessarily in the same order as the defs.
; PR13742
define i64 @swapped(i64 %x, i64 %y) nounwind {
entry:
  %x0 = call { i64, i64 } asm "foo", "=r,=r,1,0,~{dirflag},~{fpsr},~{flags}"(i64 %x, i64 %y) nounwind
  %x1 = extractvalue { i64, i64 } %x0, 0
  ret i64 %x1
}
