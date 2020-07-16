; RUN: llc < %s -mtriple=i686-pc-windows-msvc | FileCheck %s -check-prefix=X32
; Control Flow Guard is currently only available on Windows


; Test that Control Flow Guard checks are correctly added for x86 vector calls.
define void @func_cf_vector_x86(void (%struct.HVA)* %0, %struct.HVA* %1) #0 {
entry:
  %2 = alloca %struct.HVA, align 8
  %3 = bitcast %struct.HVA* %2 to i8*
  %4 = bitcast %struct.HVA* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 8 %3, i8* align 8 %4, i32 32, i1 false)
  %5 = load %struct.HVA, %struct.HVA* %2, align 8
  call x86_vectorcallcc void %0(%struct.HVA inreg %5)
  ret void

  ; X32-LABEL: func_cf_vector_x86
  ; X32: 	     movl 12(%ebp), %eax
  ; X32: 	     movl 8(%ebp), %ecx
  ; X32: 	     movups	(%eax), %xmm0
  ; X32: 	     movups	16(%eax), %xmm1
  ; X32: 	     movaps	%xmm0, (%esp)
  ; X32: 	     movaps	%xmm1, 16(%esp)
  ; X32: 	     movsd	(%esp), %xmm4
  ; X32: 	     movsd	8(%esp), %xmm5
  ; X32: 	     movsd	16(%esp), %xmm6
  ; X32: 	     movsd	24(%esp), %xmm7
  ; X32: 	     calll *___guard_check_icall_fptr
  ; X32: 	     movaps %xmm4, %xmm0
  ; X32: 	     movaps %xmm5, %xmm1
  ; X32: 	     movaps %xmm6, %xmm2
  ; X32: 	     movaps %xmm7, %xmm3
  ; X32: 	     calll  *%ecx
}
attributes #0 = { "target-cpu"="pentium4" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" }

%struct.HVA = type { double, double, double, double }

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1 immarg) #1
attributes #1 = { argmemonly nounwind willreturn }


!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
