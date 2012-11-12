; RUN: llc < %s -verify-machineinstrs -mtriple=armv7-eabi      | FileCheck %s -check-prefix=EABI
; RUN: llc < %s -verify-machineinstrs -mtriple=armv7-apple-ios -relocation-model=dynamic-no-pic | FileCheck %s -check-prefix=IOS
; RUN: llc < %s -verify-machineinstrs -mtriple=armv7-apple-ios -relocation-model=pic            | FileCheck %s -check-prefix=IOS-PIC
; RUN: llc < %s -verify-machineinstrs -mtriple=armv7-apple-ios -relocation-model=static         | FileCheck %s -check-prefix=IOS-STATIC

@foo = common global i32 0

define i32* @bar1() nounwind readnone {
entry:
; EABI:      movw    r0, :lower16:foo
; EABI-NEXT: movt    r0, :upper16:foo

; IOS:      movw    r0, :lower16:L_foo$non_lazy_ptr
; IOS-NEXT: movt    r0, :upper16:L_foo$non_lazy_ptr

; IOS-PIC:      movw    r0, :lower16:(L_foo$non_lazy_ptr-(LPC0_0+8))
; IOS-PIC-NEXT: movt    r0, :upper16:(L_foo$non_lazy_ptr-(LPC0_0+8))

; IOS-STATIC-NOT:      movw    r0, :lower16:_foo
; IOS-STATIC-NOT:       movt    r0, :upper16:_foo
  ret i32* @foo
}

define void @bar2(i32 %baz) nounwind {
entry:
; EABI:      movw    r1, :lower16:foo
; EABI-NEXT: movt    r1, :upper16:foo

; IOS:      movw    r1, :lower16:L_foo$non_lazy_ptr
; IOS-NEXT: movt    r1, :upper16:L_foo$non_lazy_ptr

; IOS-PIC:      movw    r1, :lower16:(L_foo$non_lazy_ptr-(LPC1_0+8))
; IOS-PIC-NEXT: movt    r1, :upper16:(L_foo$non_lazy_ptr-(LPC1_0+8))

; IOS-STATIC-NOT:      movw    r1, :lower16:_foo
; IOS-STATIC-NOT:      movt    r1, :upper16:_foo
  store i32 %baz, i32* @foo, align 4
  ret void
}
