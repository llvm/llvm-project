; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f() #0 {
  ret void
}

define void @g() #1 {
  ret void
}

define void @h() #2 {
  ret void
}

attributes #0 = {
; CHECK:  invalid value for 'sign-return-address' attribute: non-loaf
  "sign-return-address"="non-loaf"
; CHECK: invalid value for 'sign-return-address-key' attribute: bad-mkey
  "sign-return-address-key"="bad-mkey"
; CHECK:   invalid value for 'branch-target-enforcement' attribute: yes-please
  "branch-target-enforcement"="yes-please" }

attributes #1 = {
; CHECK:  invalid value for 'sign-return-address' attribute: All
  "sign-return-address"="All"
; CHECK: invalid value for 'sign-return-address-key' attribute: B_Key
  "sign-return-address-key"="B_Key"
; CHECK:   invalid value for 'branch-target-enforcement' attribute: True
  "branch-target-enforcement"="True"
; CHECK:   invalid value for 'branch-protection-pauth-lr' attribute: Yes
  "branch-protection-pauth-lr"="Yes"
; CHECK:   invalid value for 'guarded-control-stack' attribute: True
  "guarded-control-stack"="True"
  }

attributes #2 = {
; CHECK:  'sign-return-address-key' present without `sign-return-address`
  "sign-return-address-key"="a_key"
  }
