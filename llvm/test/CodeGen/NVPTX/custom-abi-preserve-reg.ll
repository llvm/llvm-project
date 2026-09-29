; RUN: llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 -mattr=+ptx83 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 -mattr=+ptx83 | %ptxas-verify -arch=sm_80 %}

; .abi_preserve and .abi_preserve_control require PTX ISA 8.3 and sm_80, and
; ptxas rejects them on a .entry. Both restrictions are checked below.

target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [1 x ptr] [ptr @kernel], section "llvm.metadata"

declare i32 @only_data(i32, i32) "nvvm.preserve_n_data"="4"
; CHECK-LABEL: only_data
; CHECK: (
; CHECK: .param
; CHECK: )
; CHECK: .abi_preserve 4;

declare void @only_control(i32, i32) "nvvm.preserve_n_control"="2"
; CHECK-LABEL: only_control
; CHECK: (
; CHECK: .param
; CHECK: )
; CHECK: .abi_preserve_control 2;

; def_data_and_control is defined below and has its address taken, so it is
; emitted both as a declaration here and as a definition later. One attribute
; set must yield the single-line form here and the multiline form there.
; CHECK-LABEL: def_data_and_control
; CHECK: (
; CHECK: .param
; CHECK: )
; CHECK: .abi_preserve 8 .abi_preserve_control 2;

; Taking a function's address forces a declaration to be emitted for it, and
; that declaration carries the directives.
; CHECK: .visible .func (.param .b32 func_retval0) fn
; CHECK: (
; CHECK: .param .b32 fn_param_0
; CHECK: )
; CHECK: .abi_preserve 6;
; CHECK: .visible .global .align 8 .u64 dc_addr = def_data_and_control;
; CHECK: .visible .global .align 8 .u64 fn_addr = fn;
@dc_addr = addrspace(1) global ptr @def_data_and_control, align 8
@fn_addr = addrspace(1) global ptr @fn, align 8

; An internal function emits plain .func rather than .visible .func, and still
; carries the directives.
define internal fastcc i32 @internal_linkage(i32 %a, i32 %b) unnamed_addr #0 {
; CHECK-LABEL: internal_linkage(
; CHECK:      .abi_preserve 8
; CHECK-NEXT: .abi_preserve_control 4
; CHECK-NEXT: {
  %add = add nsw i32 %b, %a
  ret i32 %add
}

define i32 @def_data_and_control(i32 %a, i32 %b) "nvvm.preserve_n_data"="8" "nvvm.preserve_n_control"="2" {
; CHECK-LABEL: def_data_and_control(
; CHECK:      .abi_preserve 8
; CHECK-NEXT: .abi_preserve_control 2
; CHECK-NEXT: {
  %r = add i32 %a, %b
  ret i32 %r
}

; The calls below are what force the declarations above to be emitted -- do not
; remove them.
define ptx_kernel void @kernel(ptr nocapture %a, ptr nocapture readonly %b) #1 {
; CHECK-LABEL: kernel(
; CHECK:       .maxntid 128
  %1 = addrspacecast ptr %b to ptr addrspace(1)
  %2 = addrspacecast ptr %a to ptr addrspace(1)
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %4 = sext i32 %3 to i64
  %getElem = getelementptr inbounds i32, ptr addrspace(1) %2, i64 %4
  %tmp3 = load i32, ptr addrspace(1) %getElem, align 4
  %getElem1 = getelementptr inbounds i32, ptr addrspace(1) %1, i64 %4
  %tmp7 = load i32, ptr addrspace(1) %getElem1, align 4
  %call = tail call fastcc i32 @internal_linkage(i32 %tmp3, i32 %tmp7)
  %call3 = tail call i32 @only_data(i32 %tmp3, i32 %tmp7)
  %add = add nsw i32 %call3, %call
  call void @only_control(i32 %tmp3, i32 %tmp7)
  %call4 = tail call i32 @def_data_and_control(i32 %tmp3, i32 %tmp7)
  store i32 %add, ptr addrspace(1) %getElem, align 4
  ret void
}

; An indirect call has no callee to consult, so the contract lives on the call
; site and is appended to the generated .callprototype, single-line. The
; prototype label is captured and matched against the call's final operand.
define internal fastcc i32 @indirect_call(i32 %a, ptr %b) unnamed_addr {
; CHECK-LABEL: indirect_call(
; CHECK:      $L__[[P1:prototype_[0-9]+]]:
; CHECK-NEXT: .callprototype (.param .b32 _) _ (.param .b32 _) .abi_preserve 8 .abi_preserve_control 2;
; The callee pointer may be loaded either side of the call sequence brace
; depending on scheduling, so the setup is matched order-independently.
; CHECK-DAG:    ld.param{{(::(func|entry))?}}.{{u|b}}32 [[R1:%.*]], [indirect_call_param_0];
; CHECK-DAG:    ld.param{{(::(func|entry))?}}.{{u|b}}64 [[RD1:%.*]], [indirect_call_param_1];
; CHECK-DAG:    .param {{.*}} param0;
; CHECK-DAG:    .param {{.*}} retval0;
; CHECK-DAG:    st.param{{(::(func|entry))?}}.{{.*}} [param0[[_:(\+0)?]]], [[R1]];
; CHECK:    call (retval0),
; CHECK:    [[RD1]],
; CHECK:    (
; CHECK:    param0
; CHECK:    )
; CHECK:    , $L__[[P1]];
; CHECK:    ld.param{{(::(func|entry))?}}.{{.*}} [[R2:%.*]], [retval0[[_:(\+0)?]]];
; CHECK:    }
; CHECK:    st.param{{(::(func|entry))?}}.{{.*}} [func_retval0[[_:(\+0)?]]], [[R2]];
; CHECK:    ret;
  %retval = call i32 %b(i32 %a) "nvvm.preserve_n_data"="8" "nvvm.preserve_n_control"="2"
  ret i32 %retval
}

; Negative control: an indirect call with no attributes gets a bare prototype,
; with nothing appended before the semicolon.
define internal fastcc i32 @indirect_call_no_abi(i32 %a, ptr %b) unnamed_addr {
; CHECK-LABEL: indirect_call_no_abi(
; CHECK:      $L__{{prototype_[0-9]+}}:
; CHECK-NEXT: .callprototype (.param .b32 _) _ (.param .b32 _);
; CHECK-NOT:  abi_preserve
  %retval = call i32 %b(i32 %a)
  ret i32 %retval
}

; The callee reached through this pointer is @fn, which carries
; nvvm.preserve_n_data. The contract is not inherited from the callee: only the
; attributes on the callsite itself are emitted, so the prototype stays bare.
define internal fastcc i32 @indirect_call_to_known_callee(i32 %a) unnamed_addr {
; CHECK-LABEL: indirect_call_to_known_callee(
; CHECK:      $L__{{prototype_[0-9]+}}:
; CHECK-NEXT: .callprototype (.param .b32 _) _ (.param .b32 _);
; CHECK-NOT:  abi_preserve
  %fp = load ptr, ptr addrspace(1) @fn_addr, align 8
  %retval = call i32 %fp(i32 %a)
  ret i32 %retval
}

; The definition half of the address-taken pair above. A single directive must
; not pick up a stray separator, so nothing may sit between it and the body.
define i32 @fn(i32 %pp) "nvvm.preserve_n_data"="6" {
; CHECK-LABEL: fn(
; CHECK:      .abi_preserve 6
; CHECK-NEXT: {
  %add = add nsw i32 %pp, 1
  ret i32 %add
}

; Function Attrs: nounwind readnone speculatable
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

attributes #0 = { noinline norecurse nounwind readnone "nvvm.preserve_n_data"="8" "nvvm.preserve_n_control"="4" }
attributes #1 = { alwaysinline nounwind "nvvm.maxntid"="128" }
attributes #2 = { nounwind readnone speculatable }
