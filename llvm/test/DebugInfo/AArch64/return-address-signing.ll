; RUN: llc -mtriple=aarch64-arm-none-eabi < %s -filetype=obj -o - \
; RUN:    | llvm-dwarfdump -v - | FileCheck -check-prefix=CHECK %s

;CHECK: CIE
;CHECK: Augmentation:          "zR"
define i32 @foo()  "sign-return-address"="all" {
  ret i32 0
}

;CHECK: CIE
;CHECK: Augmentation:          "zRB"

define i32 @bar()  "sign-return-address"="all" "sign-return-address-key"="b_key" {
  ret i32 0
}

;CHECK-NOT: CIE

define i32 @baz()  "sign-return-address"="all" nounwind {
  ret i32 0
}

;CHECK-NOT: CIE

define i32 @qux()  "sign-return-address"="all" "sign-return-address-key"="b_key" nounwind {
  ret i32 0
}
