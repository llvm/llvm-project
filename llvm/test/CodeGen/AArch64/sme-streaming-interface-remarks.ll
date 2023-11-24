; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme,+sve -verify-machineinstrs --pass-remarks-analysis=sme -o /dev/null < %s 2>&1 | FileCheck %s

declare void @normal_callee()
declare void @streaming_callee() "aarch64_pstate_sm_enabled"
declare void @streaming_compatible_callee() "aarch64_pstate_sm_compatible"

; CHECK: remark: <unknown>:0:0: call from 'normal_caller_streaming_callee' to 'streaming_callee' requires a streaming mode transition
define void @normal_caller_streaming_callee() nounwind {
  call void @streaming_callee()
  ret void;
}

; CHECK: remark: <unknown>:0:0: call from 'streaming_caller_normal_callee' to 'normal_callee' requires a streaming mode transition
define void @streaming_caller_normal_callee() nounwind "aarch64_pstate_sm_enabled" {
  call void @normal_callee()
  ret void;
}

; CHECK-NOT: streaming_caller_streaming_callee
define void @streaming_caller_streaming_callee() nounwind "aarch64_pstate_sm_enabled" {
  call void @streaming_callee()
  ret void;
}

; CHECK-NOT: streaming_caller_streaming_compatible_callee
define void @streaming_caller_streaming_compatible_callee() nounwind "aarch64_pstate_sm_enabled" {
  call void @streaming_compatible_callee()
  ret void;
}

; CHECK: remark: <unknown>:0:0: call from 'call_to_function_pointer_streaming_enabled' to 'unknown callee' requires a streaming mode transition
define void @call_to_function_pointer_streaming_enabled(ptr %p) nounwind {
  call void %p() "aarch64_pstate_sm_enabled"
  ret void
}

; CHECK: remark: <unknown>:0:0: call from 'smstart_clobber_simdfp' to 'streaming_callee' requires a streaming mode transition
define <4 x i32> @smstart_clobber_simdfp(<4 x i32> %x) nounwind {
  call void @streaming_callee()
  ret <4 x i32> %x;
}

; CHECK: remark: <unknown>:0:0: call from 'smstart_clobber_sve' to 'streaming_callee' requires a streaming mode transition
define <vscale x 4 x i32> @smstart_clobber_sve(<vscale x 4 x i32> %x) nounwind {
  call void @streaming_callee()
  ret <vscale x 4 x i32> %x;
}

; CHECK: remark: <unknown>:0:0: call from 'smstart_clobber_sve_duplicate' to 'streaming_callee' requires a streaming mode transition
; CHECK: remark: <unknown>:0:0: call from 'smstart_clobber_sve_duplicate' to 'streaming_callee' requires a streaming mode transition
define <vscale x 4 x i32> @smstart_clobber_sve_duplicate(<vscale x 4 x i32> %x) nounwind {
  call void @streaming_callee()
  call void @streaming_callee()
  ret <vscale x 4 x i32> %x;
}

; CHECK: remark: <unknown>:0:0: call from 'call_to_intrinsic_without_chain' to 'cos' requires a streaming mode transition
define double @call_to_intrinsic_without_chain(double %x) nounwind "aarch64_pstate_sm_enabled" {
entry:
  %res = call fast double @llvm.cos.f64(double %x)
  %res.fadd = fadd fast double %res, %x
  ret double %res.fadd
}

declare double @llvm.cos.f64(double)

; CHECK: remark: <unknown>:0:0: call from 'disable_tailcallopt' to 'streaming_callee' requires a streaming mode transition
define void @disable_tailcallopt() nounwind {
  tail call void @streaming_callee()
  ret void;
}

; CHECK: remark: <unknown>:0:0: call from 'call_to_non_streaming_pass_sve_objects' to 'foo' requires a streaming mode transition
define i8 @call_to_non_streaming_pass_sve_objects(ptr nocapture noundef readnone %ptr) #0 {
entry:
  %Data1 = alloca <vscale x 16 x i8>, align 16
  %Data2 = alloca <vscale x 16 x i8>, align 16
  %Data3 = alloca <vscale x 16 x i8>, align 16
  %0 = tail call i64 @llvm.aarch64.sme.cntsb()
  call void @foo(ptr noundef nonnull %Data1, ptr noundef nonnull %Data2, ptr noundef nonnull %Data3, i64 noundef %0)
  %1 = load <vscale x 16 x i8>, ptr %Data1, align 16
  %vecext = extractelement <vscale x 16 x i8> %1, i64 0
  ret i8 %vecext
}

declare i64 @llvm.aarch64.sme.cntsb()

declare void @foo(ptr noundef, ptr noundef, ptr noundef, i64 noundef)

attributes #0 = { nounwind vscale_range(1,16) "aarch64_pstate_sm_enabled" }
