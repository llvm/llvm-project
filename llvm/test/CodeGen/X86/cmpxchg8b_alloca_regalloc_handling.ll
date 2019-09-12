; RUN: llc < %s -mtriple=i686-- -stackrealign -O2 | FileCheck %s
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

; We used to have a bug when combining:
; - base pointer for stack frame (VLA + alignment)
; - cmpxchg8b frameindex + index reg

declare void @escape(i32*)

define void @foo_alloca_index(i32 %i, i64 %val) {
entry:
  %Counters = alloca [19 x i64], align 32
  %vla = alloca i32, i32 %i
  call void @escape(i32* %vla)
  br label %body

body:
  %p = getelementptr inbounds [19 x i64], [19 x i64]* %Counters, i32 0, i32 %i
  %t2 = cmpxchg volatile i64* %p, i64 %val, i64 %val seq_cst seq_cst
  %t3 = extractvalue { i64, i1 } %t2, 0
  %cmp.i = icmp eq i64 %val, %t3
  br i1 %cmp.i, label %done, label %body

done:
  ret void
}

; Check that we add a LEA
; CHECK-LABEL: foo_alloca_index:
; CHECK: leal    {{[0-9]*\(%e..,%e..,8\), %e..}}
; CHECK: lock            cmpxchg8b       ({{%e..}})



; We used to have a bug when combining:
; - base pointer for stack frame (VLA + alignment)
; - cmpxchg8b global + index reg

@Counters = external global [19 x i64]

define void @foo_alloca_index_global(i32 %i, i64 %val) {
entry:
  %aligner = alloca i32, align 32
  call void @escape(i32* %aligner)
  %vla = alloca i32, i32 %i
  call void @escape(i32* %vla)
  br label %body

body:
  %p = getelementptr inbounds [19 x i64], [19 x i64]* @Counters, i32 0, i32 %i
  %t2 = cmpxchg volatile i64* %p, i64 %val, i64 %val seq_cst seq_cst
  %t3 = extractvalue { i64, i1 } %t2, 0
  %cmp.i = icmp eq i64 %val, %t3
  br i1 %cmp.i, label %done, label %body

done:
  ret void
}

; Check that we add a LEA
; CHECK-LABEL: foo_alloca_index_global:
; CHECK: leal    {{Counters\(,%e..,8\), %e..}}
; CHECK: lock            cmpxchg8b       ({{%e..}})
