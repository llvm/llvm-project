; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; TLS function were wrongly model and after fixing that, shrink-wrapping
; cannot help here. To achieve the expected lowering, we need to playing
; tricks similar to AArch64 fast TLS calling convention (r255821).
; Applying tricks on x86-64 similar to r255821.
; RUN: llc < %s -mtriple=x86_64-apple-darwin -enable-shrink-wrap=true | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -O0 | FileCheck %s --check-prefix=CHECK-O0
%struct.S = type { i8 }

@sg = internal thread_local global %struct.S zeroinitializer, align 1
@__dso_handle = external global i8
@__tls_guard = internal thread_local unnamed_addr global i1 false
@sum1 = internal thread_local global i32 0, align 4

declare void @_ZN1SC1Ev(%struct.S*)
declare void @_ZN1SD1Ev(%struct.S*)
declare i32 @_tlv_atexit(void (i8*)*, i8*, i8*)

; Every GPR should be saved - except rdi, rax, and rsp
; CHECK-LABEL: _ZTW2sg
; CHECK-NOT: pushq %r11
; CHECK-NOT: pushq %r10
; CHECK-NOT: pushq %r9
; CHECK-NOT: pushq %r8
; CHECK-NOT: pushq %rsi
; CHECK-NOT: pushq %rdx
; CHECK-NOT: pushq %rcx
; CHECK-NOT: pushq %rbx
; CHECK: callq
; CHECK: jne
; CHECK: callq
; CHECK: tlv_atexit
; CHECK: callq
; CHECK-NOT: popq %rbx
; CHECK-NOT: popq %rcx
; CHECK-NOT: popq %rdx
; CHECK-NOT: popq %rsi
; CHECK-NOT: popq %r8
; CHECK-NOT: popq %r9
; CHECK-NOT: popq %r10
; CHECK-NOT: popq %r11
define cxx_fast_tlscc nonnull %struct.S* @_ZTW2sg() nounwind {
  %.b.i = load i1, i1* @__tls_guard, align 1
  br i1 %.b.i, label %__tls_init.exit, label %init.i

init.i:
  store i1 true, i1* @__tls_guard, align 1
  tail call void @_ZN1SC1Ev(%struct.S* nonnull @sg) #2
  %1 = tail call i32 @_tlv_atexit(void (i8*)* nonnull bitcast (void (%struct.S*)* @_ZN1SD1Ev to void (i8*)*), i8* nonnull getelementptr inbounds (%struct.S, %struct.S* @sg, i64 0, i32 0), i8* nonnull @__dso_handle) #2
  br label %__tls_init.exit

__tls_init.exit:
  ret %struct.S* @sg
}

; CHECK-LABEL: _ZTW4sum1
; CHECK-NOT: pushq %r11
; CHECK-NOT: pushq %r10
; CHECK-NOT: pushq %r9
; CHECK-NOT: pushq %r8
; CHECK-NOT: pushq %rsi
; CHECK-NOT: pushq %rdx
; CHECK-NOT: pushq %rcx
; CHECK-NOT: pushq %rbx
; CHECK: callq
define cxx_fast_tlscc nonnull i32* @_ZTW4sum1() nounwind {
  ret i32* @sum1
}

; Make sure at O0 we don't overwrite RBP.
; CHECK-O0-LABEL: _ZTW4sum2
; CHECK-O0: pushq %rbp
; CHECK-O0: movq %rsp, %rbp
; CHECK-O0-NOT: movq %r{{.*}}, (%rbp) 
define cxx_fast_tlscc i32* @_ZTW4sum2() #0 {
  ret i32* @sum1
}

attributes #0 = { nounwind "no-frame-pointer-elim"="true" }
