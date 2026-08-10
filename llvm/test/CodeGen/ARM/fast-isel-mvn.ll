; RUN: llc -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mattr=+no-movt -mtriple=armv7-apple-ios     < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM
; RUN: llc -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mattr=+no-movt -mtriple=armv7-linux-gnueabi < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM
; RUN: llc -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mattr=+no-movt -mtriple=thumbv7-apple-ios   < %s | FileCheck %s --check-prefix=CHECK --check-prefix=ARM
; RUN: llc -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios   < %s | FileCheck %s --check-prefix=CHECK --check-prefix=THUMB
; RUN: llc -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios     < %s | FileCheck %s --check-prefix=MOVT
; rdar://10412592

define void @t1() nounwind {
entry:
; CHECK-LABEL: t1
; CHECK:       mvn r0, #0
  call void @foo(i32 -1)
  ret void
}

declare void @foo(i32)

define void @t2() nounwind {
entry:
; CHECK-LABEL: t2
; CHECK:       mvn r0, #233
  call void @foo(i32 -234)
  ret void
}

define void @t3() nounwind {
entry:
; CHECK-LABEL: t3
; CHECK:       mvn r0, #256
  call void @foo(i32 -257)
  ret void
}

; Load from constant pool
define void @t4() nounwind {
entry:
; ARM-LABEL:   t4
; ARM:         ldr r0
; THUMB-LABEL: t4
; THUMB:       movw r0, #65278
; THUMB:       movt r0, #65535
  call void @foo(i32 -258)
  ret void
}

define void @t5() nounwind {
entry:
; CHECK-LABEL: t5
; CHECK:       mvn r0, #65280
  call void @foo(i32 -65281)
  ret void
}

define void @t6() nounwind {
entry:
; CHECK-LABEL: t6
; CHECK:       mvn r0, #978944
  call void @foo(i32 -978945)
  ret void
}

define void @t7() nounwind {
entry:
; CHECK-LABEL: t7
; CHECK:       mvn r0, #267386880
  call void @foo(i32 -267386881)
  ret void
}

define void @t8() nounwind {
entry:
; CHECK-LABEL: t8
; CHECK:       mvn r0, #65280
  call void @foo(i32 -65281)
  ret void
}

define void @t9() nounwind {
entry:
; CHECK-LABEL: t9
; CHECK:       mvn r0, #2130706432
  call void @foo(i32 -2130706433)
  ret void
}

; Load from constant pool.
define i32 @t10(i32 %a) {
; MOVT-LABEL: t10
; MOVT:       ldr
  %1 = xor i32 -1998730207, %a
  ret i32 %1
}

