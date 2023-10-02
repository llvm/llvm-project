; RUN: llc < %s -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -fast-isel -fast-isel-abort=1 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -verify-machineinstrs --trap-unreachable | FileCheck %s
; RUN: llc < %s -fast-isel -fast-isel-abort=1 -verify-machineinstrs --trap-unreachable | FileCheck %s
; RUN: llc < %s -verify-machineinstrs --trap-unreachable --no-trap-after-noreturn | FileCheck %s
; RUN: llc < %s -fast-isel -fast-isel-abort=1 -verify-machineinstrs --trap-unreachable --no-trap-after-noreturn | FileCheck %s

target triple = "wasm32-unknown-unknown"


; Test that the LLVM trap and debug trap intrinsics are lowered to wasm
; unreachable.

declare void @llvm.trap() cold noreturn nounwind
declare void @llvm.debugtrap() nounwind

define void @trap_ret_void() {
; CHECK-LABEL: trap_ret_void:
; CHECK:         .functype trap_ret_void () -> ()
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    unreachable
; CHECK-NEXT:    # fallthrough-return
; CHECK-NEXT:    end_function
  call void @llvm.trap()
  ret void
}

define void @debugtrap_ret_void() {
; CHECK-LABEL: debugtrap_ret_void:
; CHECK:         .functype debugtrap_ret_void () -> ()
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    unreachable
; CHECK-NEXT:    # fallthrough-return
; CHECK-NEXT:    end_function
  call void @llvm.debugtrap()
  ret void
}

; LLVM trap followed by LLVM unreachable could become exactly one wasm
; unreachable, but two are emitted currently.
define void @trap_unreacheable() {
; CHECK-LABEL: trap_unreacheable:
; CHECK:         .functype trap_unreacheable () -> ()
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    unreachable
; CHECK-NEXT:    unreachable
; CHECK-NEXT:    end_function
  call void @llvm.trap()
  unreachable
}


; Test that LLVM unreachable instruction is lowered to wasm unreachable when
; necessary to fulfill the wasm operand stack requirements.

declare void @ext_func()
declare i32 @ext_func_i32()
declare void @ext_never_return() noreturn

; This test emits wasm unreachable to fill in for the missing i32 return value.
define i32 @missing_ret_unreach() {
; CHECK-LABEL: missing_ret_unreach:
; CHECK:         .functype missing_ret_unreach () -> (i32)
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    call ext_func
; CHECK-NEXT:    unreachable
; CHECK-NEXT:    end_function
  call void @ext_func()
  unreachable
}

; This is similar to the above test, but ensures wasm unreachable is emitted
; even after a noreturn call.
define i32 @missing_ret_noreturn_unreach() {
; CHECK-LABEL: missing_ret_noreturn_unreach:
; CHECK:         .functype missing_ret_noreturn_unreach () -> (i32)
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    call ext_never_return
; CHECK-NEXT:    unreachable
; CHECK-NEXT:    end_function
  call void @ext_never_return()
  unreachable
}

; We could emit no instructions at all for the llvm unreachables in these next
; three tests, as the signatures match and reaching llvm unreachable is
; undefined behaviour. But wasm unreachable is emitted for the time being.

define void @void_sig_match_unreach() {
; CHECK-LABEL: void_sig_match_unreach:
; CHECK:         .functype void_sig_match_unreach () -> ()
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    call ext_func
; CHECK-NEXT:    unreachable
; CHECK-NEXT:    end_function
  call void @ext_func()
  unreachable
}

define i32 @i32_sig_match_unreach() {
; CHECK-LABEL: i32_sig_match_unreach:
; CHECK:         .functype i32_sig_match_unreach () -> (i32)
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    call ext_func_i32
; CHECK-NEXT:    drop
; CHECK-NEXT:    unreachable
; CHECK-NEXT:    end_function
  call i32 @ext_func_i32()
  unreachable
}

define void @void_sig_match_noreturn_unreach() {
; CHECK-LABEL: void_sig_match_noreturn_unreach:
; CHECK:         .functype void_sig_match_noreturn_unreach () -> ()
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    call ext_never_return
; CHECK-NEXT:    unreachable
; CHECK-NEXT:    end_function
  call void @ext_never_return()
  unreachable
}

; This function currently doesn't emit unreachable.
define void @void_sig_match_noreturn_ret() {
; CHECK-LABEL: void_sig_match_noreturn_ret:
; CHECK:         .functype void_sig_match_noreturn_ret () -> ()
; CHECK-NEXT:  # %bb.0:
; CHECK-NEXT:    call ext_never_return
; CHECK-NEXT:    # fallthrough-return
; CHECK-NEXT:    end_function
  call void @ext_never_return()
  ret void
}
