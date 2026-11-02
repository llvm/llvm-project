; RUN: llc < %s -mtriple=arm64ec-pc-windows-msvc -o - | FileCheck %s
;
;; The 80-bit x87 `long double` occupies a 16-byte slot, but the ARM64EC side
;; passes and returns it as a pair of integer registers holding the 80-bit
;; image, while the x64 side passes it in memory (by reference) and returns it
;; through a hidden pointer.  The thunks convert between the two, moving only
;; the ten bytes that make up the value and leaving the padding alone.
;;
;;   entry thunk: x64 memory slot -> ARM64EC register pair
;;   exit thunk:  ARM64EC register pair -> x64 memory slot

target triple = "arm64ec-pc-windows-msvc"

declare x86_fp80 @ext_ret()
declare void @ext_take(x86_fp80)

; The value is an i128 in an even-aligned register pair, so having a 32-bit
; argument in x0 puts it in x2:x3, and it is returned in x1:x0.
define dso_local x86_fp80 @use_f80(i32 %x, x86_fp80 %v) {
  ret x86_fp80 %v
}

define dso_local x86_fp80 @pass_f80(x86_fp80 %v) {
  ret x86_fp80 %v
}

define dso_local void @take_f80(i32 %x, x86_fp80 %v) {
  ret void
}

; With seven integer arguments the pair no longer fits in the argument
; registers and is passed on the stack, by value on the ARM64EC side and by
; reference on the x64 side.
define dso_local void @stack_f80(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e,
                                i64 %f, i64 %g, x86_fp80 %v) {
  ret void
}

define dso_local x86_fp80 @ec_caller_ret() {
  %r = call x86_fp80 @ext_ret()
  ret x86_fp80 %r
}

define dso_local void @ec_caller_take(x86_fp80 %v) {
  call void @ext_take(x86_fp80 %v)
  ret void
}

; -----------------------------------------------------------------------------
; ARM64EC side: the argument arrives in x2:x3 and the result is returned in
; x1:x0.
; CHECK-LABEL: "#use_f80":
; CHECK:       mov     x1, x3
; CHECK-NEXT:  mov     x0, x2
; CHECK-NEXT:  ret

; -----------------------------------------------------------------------------
; Entry thunk for @use_f80: x0 is the hidden return pointer, x1 the 32-bit
; argument and x2 the pointer to the memory slot, which is loaded into the
; register pair.  The returned pair is written through the return pointer.
; CHECK-LABEL: $ientry_thunk$cdecl$__llvm_x__$i8__llvm_x__:
; CHECK:       ldr     x8, [x2]
; CHECK-NEXT:  ldrh    w3, [x2, #8]
; CHECK-NEXT:  mov     x19, x0
; CHECK-NEXT:  mov     x0, x1
; CHECK-NEXT:  mov     x2, x8
; CHECK-NEXT:  blr     x9
; CHECK:       str     x0, [x19]
; CHECK-NEXT:  strh    w1, [x19, #8]

; -----------------------------------------------------------------------------
; Entry thunk for @pass_f80: the slot pointer is in x1 and the pair fills the
; first two argument registers.
; CHECK-LABEL: $ientry_thunk$cdecl$__llvm_x__$__llvm_x__:
; CHECK:       mov     x19, x0
; CHECK-NEXT:  ldr     x0, [x1]
; CHECK-NEXT:  ldrh    w1, [x1, #8]
; CHECK-NEXT:  blr     x9
; CHECK:       str     x0, [x19]
; CHECK-NEXT:  strh    w1, [x19, #8]

; -----------------------------------------------------------------------------
; Entry thunk for @take_f80: the 32-bit argument stays in x0, so the pair goes
; into x2:x3.
; CHECK-LABEL: $ientry_thunk$cdecl$v$i8__llvm_x__:
; CHECK:       ldr     x2, [x1]
; CHECK-NEXT:  ldrh    w3, [x1, #8]
; CHECK-NEXT:  blr     x9

; -----------------------------------------------------------------------------
; Entry thunk for @stack_f80: x4 points at the x64 stack arguments, from which
; the pair is copied onto the outgoing stack, dereferencing the pointer to the
; slot that the x64 side stores there.
; CHECK-LABEL: $ientry_thunk$cdecl$v$i8i8i8i8i8i8i8__llvm_x__:
; CHECK:       ldp     x8, x5, [x4, #32]
; CHECK-NEXT:  ldr     x11, [x10]
; CHECK-NEXT:  ldrh    w10, [x10, #8]
; CHECK-NEXT:  mov     x4, x8
; CHECK-NEXT:  stp     x11, x10, [sp]
; CHECK-NEXT:  blr     x9

; -----------------------------------------------------------------------------
; Exit thunk for @ec_caller_ret: the x64 function returns through the hidden
; pointer in x0, and the slot is loaded back into the register pair.
; CHECK-LABEL: $iexit_thunk$cdecl$__llvm_x__$v:
; CHECK:       sub     x0, x29, #16
; CHECK-NEXT:  ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:  blr     x16
; CHECK-NEXT:  ldurh   w1, [x29, #-8]
; CHECK-NEXT:  ldur    x0, [x29, #-16]
; CHECK:       ret

; -----------------------------------------------------------------------------
; Exit thunk for @ec_caller_take: the incoming pair is stored to a slot whose
; address is passed to the x64 function.
; CHECK-LABEL: $iexit_thunk$cdecl$v$__llvm_x__:
; CHECK:       stur    x0, [x29, #-16]
; CHECK-NEXT:  sub     x0, x29, #16
; CHECK-NEXT:  ldr     x16, [x8, :lo12:__os_arm64x_dispatch_call_no_redirect]
; CHECK-NEXT:  sturh   w1, [x29, #-8]
; CHECK-NEXT:  blr     x16
; CHECK:       ret

; -----------------------------------------------------------------------------
; The thunks are attached to their functions in the hybrid map.
; CHECK:       .symidx "#pass_f80"
; CHECK-NEXT:  .symidx $ientry_thunk$cdecl$__llvm_x__$__llvm_x__
; CHECK:       .symidx ext_take
; CHECK-NEXT:  .symidx $iexit_thunk$cdecl$v$__llvm_x__
