; RUN: llc < %s -march=nvptx64 -mattr=+ptx64 -mcpu=sm_30 | FileCheck %s
; RUN: %if ptxas %{llc < %s -march=nvptx64 -mattr=+ptx60 -mcpu=sm_30 | %ptxas-verify %}

@function_pointer = addrspace(1) global void (i32)* null

; CHECK: .func trap_wrapper
; CHECK-NEXT: ()
; CHECK-NEXT: .noreturn;

declare void @trap_wrapper() #0

; CHECK: .func {{.*}} non_void_noreturn()
; CHECK-NOT: .noreturn

define i32 @non_void_noreturn() #0 {
  ret i32 54
}

; CHECK: .func true_noreturn0()
; CHECK-NEXT: .noreturn

define void @true_noreturn0() #0 {
  call void @trap_wrapper()
  ret void
}

; CHECK: .entry ignore_kernel_noreturn()
; CHECK-NOT: .noreturn

define void @ignore_kernel_noreturn() #0 {
  unreachable
}

; CHECK-LABEL: .entry callprototype_noreturn(
; CHECK: prototype_{{[0-9]+}} : .callprototype ()_ (.param .b32 _) .noreturn;
; CHECK: prototype_{{[0-9]+}} : .callprototype (.param .b32 _) _ (.param .b32 _);

define void @callprototype_noreturn(i32) {
  %fn = load void (i32)*, void (i32)* addrspace(1)* @function_pointer
  call void %fn(i32 %0) #0
  %non_void = bitcast void (i32)* %fn to i32 (i32)*
  %2 = call i32 %non_void(i32 %0) #0
  ret void
}

attributes #0 = { noreturn }

!nvvm.annotations = !{!0, !1}

!0 = !{void ()* @ignore_kernel_noreturn, !"kernel", i32 1}
!1 = !{void (i32)* @callprototype_noreturn, !"kernel", i32 1}
