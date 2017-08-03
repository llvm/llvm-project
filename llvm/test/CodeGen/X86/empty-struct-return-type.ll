; RUN: llc < %s -march=x86-64 | grep call
; PR4688

; Return types can be empty structs, which can be awkward.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @_ZN15QtSharedPointer22internalSafetyCheckAddEPVKv(i8* %ptr) {
entry:
	%0 = call { } @_ZNK5QHashIPv15QHashDummyValueE5valueERKS0_(i8** undef)		; <{ }> [#uses=0]
        ret void
}

declare hidden { } @_ZNK5QHashIPv15QHashDummyValueE5valueERKS0_(i8** nocapture) nounwind
