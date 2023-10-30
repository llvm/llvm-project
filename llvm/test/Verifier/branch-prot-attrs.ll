; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

define void @f() #0 {
  ret void
}

attributes #0 = {
; CHECK:  invalid value for 'sign-return-address' attribute: loaf
  "sign-return-address"="loaf"
; CHECK: invalid value for 'sign-return-address-key' attribute: bad-mkey
  "sign-return-address-key"="bad-mkey"
; CHECK:   invalid value for 'branch-target-enforcement' attribute: yes-please
  "branch-target-enforcement"="yes-please" }
