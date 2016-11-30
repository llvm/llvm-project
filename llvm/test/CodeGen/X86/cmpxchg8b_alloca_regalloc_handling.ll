; RUN: llc < %s -march=x86 -stackrealign -O2 | FileCheck %s
; PR28755

; Check that register allocator is able to handle that
; a-lot-of-fixed-and-reserved-registers case. We do that by
; emmiting lea before 4 cmpxchg8b operands generators.

define void @foo_alloca(i64* %a, i32 %off, i32 %n) {
  %dummy = alloca i32, i32 %n
  %addr = getelementptr inbounds i64, i64* %a, i32 %off

  %res = cmpxchg i64* %addr, i64 0, i64 1 monotonic monotonic
  ret void
}

; CHECK-LABEL: foo_alloca
; CHECK: leal    {{\(%e..,%e..,.*\)}}, [[REGISTER:%e.i]]
; CHECK-NEXT: xorl    %eax, %eax
; CHECK-NEXT: xorl    %edx, %edx
; CHECK-NEXT: xorl    %ecx, %ecx
; CHECK-NEXT: movl    $1, %ebx
; CHECK-NEXT: lock            cmpxchg8b       ([[REGISTER]])

; If we don't use index register in the address mode -
; check that we did not generate the lea.
define void @foo_alloca_direct_address(i64* %addr, i32 %n) {
  %dummy = alloca i32, i32 %n

  %res = cmpxchg i64* %addr, i64 0, i64 1 monotonic monotonic
  ret void
}

; CHECK-LABEL: foo_alloca_direct_address
; CHECK-NOT: leal    {{\(%e.*\)}}, [[REGISTER:%e.i]]
; CHECK: lock            cmpxchg8b       ([[REGISTER]])
