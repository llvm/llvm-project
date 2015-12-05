; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-apple-darwin -enable-shrink-wrap=true | FileCheck --check-prefix=SHRINK %s
%struct.S = type { i8 }

@sg = internal thread_local global %struct.S zeroinitializer, align 1
@__dso_handle = external global i8
@__tls_guard = internal thread_local unnamed_addr global i1 false

declare void @_ZN1SC1Ev(%struct.S*)
declare void @_ZN1SD1Ev(%struct.S*)
declare i32 @_tlv_atexit(void (i8*)*, i8*, i8*)

; Every GPR should be saved - except rdi, rax, and rsp
; CHECK-LABEL: _ZTW2sg
; CHECK: pushq %r11
; CHECK: pushq %r10
; CHECK: pushq %r9
; CHECK: pushq %r8
; CHECK: pushq %rsi
; CHECK: pushq %rdx
; CHECK: pushq %rcx
; CHECK: pushq %rbx
; CHECK: callq
; CHECK: jne
; CHECK: callq
; CHECK: tlv_atexit
; CHECK: callq
; CHECK: popq %rbx
; CHECK: popq %rcx
; CHECK: popq %rdx
; CHECK: popq %rsi
; CHECK: popq %r8
; CHECK: popq %r9
; CHECK: popq %r10
; CHECK: popq %r11
; SHRINK-LABEL: _ZTW2sg
; SHRINK: callq
; SHRINK: jne
; SHRINK: pushq %r11
; SHRINK: pushq %r10
; SHRINK: pushq %r9
; SHRINK: pushq %r8
; SHRINK: pushq %rsi
; SHRINK: pushq %rdx
; SHRINK: pushq %rcx
; SHRINK: pushq %rbx
; SHRINK: callq
; SHRINK: tlv_atexit
; SHRINK: popq %rbx
; SHRINK: popq %rcx
; SHRINK: popq %rdx
; SHRINK: popq %rsi
; SHRINK: popq %r8
; SHRINK: popq %r9
; SHRINK: popq %r10
; SHRINK: popq %r11
; SHRINK: LBB{{.*}}:
; SHRINK: callq
define cxx_fast_tlscc nonnull %struct.S* @_ZTW2sg() {
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
