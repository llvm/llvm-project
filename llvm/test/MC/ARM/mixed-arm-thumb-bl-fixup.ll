; RUN: llc -O0 < %s -mtriple armv7-linux-gnueabi -o - \
; RUN:   | llvm-mc -triple armv7-linux-gnueabi -filetype=obj -o - \
; RUN:    | llvm-readobj -r - | FileCheck --check-prefix LINUX %s

; RUN: llc -O0 < %s -mtriple armv7-linux-android -o - \
; RUN:   | llvm-mc -triple armv7-linux-android -filetype=obj -o - \
; RUN:    | llvm-readobj -r - | FileCheck --check-prefix LINUX %s


; RUN: llc -O0 < %s -mtriple armv7-apple-ios -o - \
; RUN:   | llvm-mc -triple armv7-apple-ios -filetype=obj -o - \
; RUN:    | llvm-readobj -r - | FileCheck --check-prefix IOS %s


define void @thumb_caller() #0 {
  call void @internal_arm_fn()
  call void @global_arm_fn()
  call void @internal_thumb_fn()
  call void @global_thumb_fn()
  ret void
}

define void @arm_caller() #1 {
  call void @internal_arm_fn()
  call void @global_arm_fn()
  call void @internal_thumb_fn()
  call void @global_thumb_fn()
  ret void
}

define internal void @internal_thumb_fn() #0 {
  ret void
}

define void @global_thumb_fn() #0 {
entry:
  br label %end
end:
  br label %end
  ret void
}

define internal void @internal_arm_fn() #1 {
  ret void
}

define void @global_arm_fn() #1 {
entry:
  br label %end
end:
  br label %end
  ret void
}

attributes #0 = { "target-features"="+thumb-mode" }
attributes #1 = { "target-features"="-thumb-mode" }

; LINUX: Section (3) .rel.text {
; LINUX-NEXT: 0x2 R_ARM_THM_CALL internal_arm_fn 0x0
; LINUX-NEXT: 0x6 R_ARM_THM_CALL global_arm_fn 0x0
; LINUX-NEXT: 0xE R_ARM_THM_CALL global_thumb_fn 0x0
; LINUX-NEXT: 0x1C R_ARM_CALL internal_arm_fn 0x0
; LINUX-NEXT: 0x20 R_ARM_CALL global_arm_fn 0x0
; LINUX-NEXT: 0x24 R_ARM_CALL internal_thumb_fn 0x0
; LINUX-NEXT: 0x28 R_ARM_CALL global_thumb_fn 0x0
; LINUX-NEXT: }

; IOS:   Section __text {
; IOS-NEXT: 0x2C 1 2 1 ARM_RELOC_BR24 0 _global_thumb_fn
; IOS-NEXT: 0x28 1 2 1 ARM_RELOC_BR24 0 _internal_thumb_fn
; IOS-NEXT: 0x24 1 2 1 ARM_RELOC_BR24 0 _global_arm_fn
; IOS-NEXT: 0x20 1 2 1 ARM_RELOC_BR24 0 _internal_arm_fn
; IOS-NEXT: 0x10 1 2 0 ARM_THUMB_RELOC_BR22 0 __text
; IOS-NEXT: 0xC 1 2 0 ARM_THUMB_RELOC_BR22 0 __text
; IOS-NEXT: 0x8 1 2 0 ARM_THUMB_RELOC_BR22 0 __text
; IOS-NEXT: 0x4 1 2 0 ARM_THUMB_RELOC_BR22 0 __text
; IOS-NEXT: }
