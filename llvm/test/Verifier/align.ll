; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: Attribute 'align 4' applied to incompatible type!
; CHECK-NEXT: @align_non_pointer1
define void @align_non_pointer1(i32 align 4 %a) {
  ret void
}

; CHECK: Attribute 'align 4' applied to incompatible type!
; CHECK-NEXT: @align_non_pointer2
define align 4 void @align_non_pointer2(i32 %a) {
  ret void
}
