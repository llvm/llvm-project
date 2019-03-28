; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse4.2 | FileCheck %s

; widen a v3i1 to v4i1 to do a vector load/store. We would previously
; reconstruct the said v3i1 from the first element of the vector by filling all
; the lanes of the vector with that first element, which was obviously wrong.
; This was done in the type-legalizing of the DAG, when legalizing the load.

; Function Attrs: argmemonly nounwind readonly
declare <3 x i32> @llvm.masked.load.v3i32.p1v3i32(<3 x i32> addrspace(1)*, i32, <3 x i1>, <3 x i32>)

; Function Attrs: argmemonly nounwind
declare void @llvm.masked.store.v3i32.p1v3i32(<3 x i32>, <3 x i32> addrspace(1)*, i32, <3 x i1>)

define  <3 x i32> @masked_load_v3(i32 addrspace(1)*, <3 x i1>) {
entry:
  %2 = bitcast i32 addrspace(1)* %0 to <3 x i32> addrspace(1)*
  %3 = call <3 x i32> @llvm.masked.load.v3i32.p1v3i32(<3 x i32> addrspace(1)* %2, i32 4, <3 x i1> %1, <3 x i32> undef)
  ret <3 x i32> %3
}

define void @masked_store4_v3(<3 x i32>, i32 addrspace(1)*, <3 x i1>) {
entry:
  %3 = bitcast i32 addrspace(1)* %1 to <3 x i32> addrspace(1)*
  call void @llvm.masked.store.v3i32.p1v3i32(<3 x i32> %0, <3 x i32> addrspace(1)* %3, i32 4, <3 x i1> %2)
  ret void
}

define void @local_load_v3i1(i32 addrspace(1)* %out, i32 addrspace(1)* %in, <3 x i1>* %predicate_ptr) nounwind {
; CHECK-LABEL: local_load_v3i1:
; CHECK:       # %bb.0:
; CHECK-NEXT: pushq   %rbp
; CHECK-NEXT: pushq   %r15
; CHECK-NEXT: pushq   %r14
; CHECK-NEXT: pushq   %rbx
; CHECK-NEXT: pushq   %rax
; CHECK-NEXT: movq    %rdi, %r14
; CHECK-NEXT: movzbl  (%rdx), %ebp
; CHECK-NEXT: movl    %ebp, %eax
; CHECK-NEXT: shrl    %eax
; CHECK-NEXT: andl    $1, %eax
; CHECK-NEXT: movl    %ebp, %ecx
; CHECK-NEXT: andl    $1, %ecx
; CHECK-NEXT: movd    %ecx, %xmm0
; CHECK-NEXT: pinsrd  $1, %eax, %xmm0
; CHECK-NEXT: shrl    $2, %ebp
; CHECK-NEXT: andl    $1, %ebp
; CHECK-NEXT: pinsrd  $2, %ebp, %xmm0
; CHECK-NEXT: movd    %xmm0, %ebx
; CHECK-NEXT: pextrd  $1, %xmm0, %r15d
; CHECK-NEXT: movq    %rsi, %rdi
; CHECK-NEXT: movl    %ebx, %esi
; CHECK-NEXT: movl    %r15d, %edx
; CHECK-NEXT: movl    %ebp, %ecx
; CHECK-NEXT: callq   masked_load_v3
; CHECK-NEXT: movq    %r14, %rdi
; CHECK-NEXT: movl    %ebx, %esi
; CHECK-NEXT: movl    %r15d, %edx
; CHECK-NEXT: movl    %ebp, %ecx
; CHECK-NEXT: callq   masked_store4_v3
; CHECK-NEXT: addq    $8, %rsp
; CHECK-NEXT: popq    %rbx
; CHECK-NEXT: popq    %r14
; CHECK-NEXT: popq    %r15
; CHECK-NEXT: popq    %rbp
; CHECK-NEXT: retq
  %predicate = load <3 x i1>, <3 x i1>* %predicate_ptr
  %load1 = call <3 x i32> @masked_load_v3(i32 addrspace(1)* %in, <3 x i1> %predicate)
  call void @masked_store4_v3(<3 x i32> %load1, i32 addrspace(1)* %out, <3 x i1> %predicate)
  ret void
}
