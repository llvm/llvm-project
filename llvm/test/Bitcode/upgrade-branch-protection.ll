;; Test that module flags "branch-target-enforcement" and "sign-return-address"
;; can be upgraded to are upgraded from Error to Min and the value is changed 2
;; as the module is converted to the semantic.

; RUN: llvm-as %s -o - | llvm-dis - | FileCheck %s
; RUN: %if asserts %{ llvm-as %s -debug-only=auto-upgrade -o /dev/null 2>&1 | FileCheck %s --check-prefix CHECK-LOG %}

target triple = "aarch64-unknown-linux-gnu"

define i32 @foo_on() #0 {
entry:
  ret i32 42
}

define i32 @foo_off() #1 {
entry:
  ret i32 43
}

attributes #0 = { noinline nounwind optnone uwtable "branch-target-enforcement"="true"}
attributes #1 = { noinline nounwind optnone uwtable "branch-target-enforcement"="false" "sign-return-address"="none" }

!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"branch-target-enforcement", i32 1}
!1 = !{i32 1, !"sign-return-address", i32 1}
!2 = !{i32 1, !"sign-return-address-all", i32 1}
!3 = !{i32 1, !"sign-return-address-with-bkey", i32 1}

;CHECK: define{{.+}}@foo_on{{.+}}#[[ATTR_ON:[0-9]+]]
;CHECK: define{{.+}}@foo_off{{.+}}#[[ATTR_OFF:[0-9]+]]

;CHECK: attributes #[[ATTR_ON]] {{.+}}"branch-target-enforcement" "sign-return-address"="all" "sign-return-address-key"="b_key"
;CHECK: attributes #[[ATTR_OFF]] {{.+}}"sign-return-address"="none" "sign-return-address-key"="b_key"

;CHECK: !0 = !{i32 8, !"branch-target-enforcement", i32 2}
;CHECK: !1 = !{i32 8, !"sign-return-address", i32 2}
;CHECK: !2 = !{i32 8, !"sign-return-address-all", i32 2}
;CHECK: !3 = !{i32 8, !"sign-return-address-with-bkey", i32 2}

;CHECK-LOG: Found module flag: branch-target-enforcement(1)
;CHECK-LOG: Found module flag: sign-return-address(1)
;CHECK-LOG: Found module flag: sign-return-address-all(1)
;CHECK-LOG: Found module flag: sign-return-address-with-bkey(1)
;CHECK-LOG: Set attribute: sign-return-address="all", function: foo_on
;CHECK-LOG: Set attribute: sign-return-address-key="b_key", function: foo_on
;CHECK-LOG: Converted attribute: branch-target-enforcement="true", function: foo_on
;CHECK-LOG: Set attribute: sign-return-address-key="b_key", function: foo_off
;CHECK-LOG: Removed attribute: branch-target-enforcement="false", function: foo_off
;CHECK-LOG: Converted module flag: {8, branch-target-enforcement, 2}
;CHECK-LOG: Converted module flag: {8, sign-return-address, 2}
;CHECK-LOG: Converted module flag: {8, sign-return-address-all, 2}
;CHECK-LOG: Converted module flag: {8, sign-return-address-with-bkey, 2}
