; RUN: llc < %s -mtriple=i686-pc-windows-msvc | FileCheck %s -check-prefix=X32
; Control Flow Guard is currently only available on Windows


; Test that Control Flow Guard checks are correctly added for x86 vector calls.
define void @func_cf_vector_x86(ptr %0, ptr %1) #0 {
entry:
  %2 = alloca %struct.HVA, align 8
  call void @llvm.memcpy.p0.p0.i32(ptr align 8 %2, ptr align 8 %1, i32 32, i1 false)
  %3 = load %struct.HVA, ptr %2, align 8
  call x86_vectorcallcc void %0(%struct.HVA inreg %3)
  ret void

  ; X32-LABEL: func_cf_vector_x86
  ; X32: 	     movl 12(%ebp), %eax
  ; X32: 	     movl 8(%ebp), %ecx
  ; X32: 	     movsd 24(%eax), %xmm4         # xmm4 = mem[0],zero
  ; X32: 	     movsd %xmm4, 24(%esp)
  ; X32: 	     movsd 16(%eax), %xmm5         # xmm5 = mem[0],zero
  ; X32: 	     movsd %xmm5, 16(%esp)
  ; X32: 	     movsd (%eax), %xmm6           # xmm6 = mem[0],zero
  ; X32: 	     movsd 8(%eax), %xmm7          # xmm7 = mem[0],zero
  ; X32: 	     movsd %xmm7, 8(%esp)
  ; X32: 	     movsd %xmm6, (%esp)
  ; X32: 	     calll *___guard_check_icall_fptr
  ; X32: 	     movaps %xmm6, %xmm0
  ; X32: 	     movaps %xmm7, %xmm1
  ; X32: 	     movaps %xmm5, %xmm2
  ; X32: 	     movaps %xmm4, %xmm3
  ; X32: 	     calll  *%ecx
}
attributes #0 = { "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }

%struct.HVA = type { double, double, double, double }

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1 immarg) #1
attributes #1 = { argmemonly nounwind willreturn }


!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
