; RUN: llc -o - %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; Check that we do global variable demotion when the symbols don't need to be
; shared across modules or functions.

; Variable visible globally can't be demoted.
; CHECK: .visible .shared .align 4 .u64 external_global
@external_global = addrspace(3) global i64 undef, align 4

; Not externally visible global used in only one function => demote.
; It should live in @define_internal_global.
@internal_global = internal addrspace(3) global i64 undef, align 4

; Not externally visible global used in several functions => keep.
; CHECK: .shared .align 4 .u64 internal_global_used_in_different_fcts
@internal_global_used_in_different_fcts = internal addrspace(3) global i64 undef, align 4

; Not externally visible global used in only one function => demote.
; It should live in @define_private_global.
@private_global = private addrspace(3) global i64 undef, align 4

; Not externally visible global used in only one function
; (but with several uses) => demote.
; It should live in @define_private_global_more_than_one_use.
@private_global_used_more_than_once_in_same_fct = private addrspace(3) global i64 undef, align 4

; CHECK-LABEL: define_external_global(
define void @define_external_global(i64 %val) {
  store i64 %val, ptr addrspace(3) @external_global
  ret void
}

; CHECK-LABEL: define_internal_global(
; Demoted `internal_global` should live here.
; CHECK: .shared .align 4 .u64 internal_global
define void @define_internal_global(i64 %val) {
  store i64 %val, ptr addrspace(3) @internal_global
  ret void
}

; CHECK-LABEL: define_internal_global_with_different_fct1(
; CHECK-NOT: .shared .align 4 .u64 internal_global_used_in_different_fcts
define void @define_internal_global_with_different_fct1(i64 %val) {
  store i64 %val, ptr addrspace(3) @internal_global_used_in_different_fcts
  ret void
}

; CHECK-LABEL: define_internal_global_with_different_fct2(
; CHECK-NOT: .shared .align 4 .u64 internal_global_used_in_different_fcts
define void @define_internal_global_with_different_fct2(i64 %val) {
  store i64 %val, ptr addrspace(3) @internal_global_used_in_different_fcts
  ret void
}

; CHECK-LABEL: define_private_global(
; Demoted `private_global` should live here.
; CHECK: .shared .align 4 .u64 private_global
define void @define_private_global(i64 %val) {
  store i64 %val, ptr addrspace(3) @private_global
  ret void
}

; CHECK-LABEL: define_private_global_more_than_one_use(
; Demoted `private_global_used_more_than_once_in_same_fct` should live here.
; CHECK: .shared .align 4 .u64 private_global_used_more_than_once_in_same_fct
;
; Also check that the if-then is still here, otherwise we may not be testing
; the "more-than-one-use" part.
; CHECK: st.shared.u64   [private_global_used_more_than_once_in_same_fct],
; CHECK: mov.u64 %[[VAR:.*]], 25
; CHECK: st.shared.u64   [private_global_used_more_than_once_in_same_fct], %[[VAR]]
define void @define_private_global_more_than_one_use(i64 %val, i1 %cond) {
  store i64 %val, ptr addrspace(3) @private_global_used_more_than_once_in_same_fct
  br i1 %cond, label %then, label %end

then:
  store i64 25, ptr addrspace(3) @private_global_used_more_than_once_in_same_fct
  br label %end

end:
  ret void
}
