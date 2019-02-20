; RUN: llc --code-model=kernel <%s -asm-verbose=0 | FileCheck %s
; PR4933
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
%struct.kmem_cache_order_objects = type { i64 }
declare i8* @memset(i8*, i32, i64)
define void @unxlate_dev_mem_ptr(i64 %phis, i8* %addr) nounwind {
  %pte.addr.i = alloca %struct.kmem_cache_order_objects*
  %call8 = call i8* @memset(i8* bitcast ([512 x %struct.kmem_cache_order_objects]* @bm_pte to i8*), i32 0, i64 4096)
; CHECK:      movl    $4096, %edx
; CHECK-NEXT: movq    $bm_pte, %rdi
; CHECK-NEXT: xorl    %esi, %esi
; CHECK-NEXT: callq   memset
  ret void
}
@bm_pte = internal global [512 x %struct.kmem_cache_order_objects] zeroinitializer, section ".bss.page_aligned", align 4096
; CHECK: .section        .bss.page_aligned,"aw",@nobits
; CHECK-NEXT: .p2align  12
; CHECK-NEXT: bm_pte:
; CHECK-NEXT: .zero   4096
; CHECK-NEXT: .size   bm_pte, 4096
