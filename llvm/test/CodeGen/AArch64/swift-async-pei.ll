; RUN: llc -mtriple arm64-apple-ios -filetype asm -o - %s -swift-async-fp always | FileCheck %s -check-prefix CHECK-IOS-ALWAYS
; RUN: llc -mtriple arm64-apple-ios -filetype asm -o - %s -swift-async-fp never | FileCheck %s -check-prefix CHECK-IOS-NEVER
; RUN: llc -mtriple arm64-apple-ios -filetype asm -o - %s -swift-async-fp auto | FileCheck %s -check-prefix CHECK-IOS-AUTO
; RUN: llc -mtriple arm64_32-apple-watchos -filetype asm -o - %s -swift-async-fp always | FileCheck %s -check-prefix CHECK-WATCHOS-ALWAYS
; RUN: llc -mtriple arm64_32-apple-watchos -filetype asm -o - %s -swift-async-fp never | FileCheck %s -check-prefix CHECK-WATCHOS-NEVER
; RUN: llc -mtriple arm64_32-apple-watchos -filetype asm -o - %s -swift-async-fp auto | FileCheck %s -check-prefix CHECK-WATCHOS-AUTO

declare ptr @llvm.swift.async.context.addr()

define swifttailcc void @f(ptr swiftasync %ctx) {
  %1 = load ptr, ptr %ctx, align 8
  %2 = tail call ptr @llvm.swift.async.context.addr()
  store ptr %1, ptr %2, align 8
  ret void
}

; CHECK-IOS-NEVER: sub sp, sp, #32
; CHECK-IOS-NEVER: stp x29, x30, [sp, #16]
; ...
; CHECK-IOS-NEVER: ldp x29, x30, [sp, #16]
; CHECK-IOS-NEVER: add sp, sp, #32

; CHECK-IOS-ALWAYS: orr x29, x29, #0x1000000000000000
; CHECK-IOS-ALWAYS: sub sp, sp, #32
; CHECK-IOS-ALWAYS: stp x29, x30, [sp, #16]
; ...
; CHECK-IOS-ALWAYS: ldp x29, x30, [sp, #16]
; CHECK-IOS-ALWAYS: and x29, x29, #0xefffffffffffffff
; CHECK-IOS-ALWAYS: add sp, sp, #32

; CHECK-IOS-AUTO: adrp x16, _swift_async_extendedFramePointerFlags@GOTPAGE
; CHECK-IOS-AUTO: ldr x16, [x16, _swift_async_extendedFramePointerFlags@GOTPAGEOFF]
; CHECK-IOS-AUTO: orr x29, x29, x16
; CHECK-IOS-AUTO: sub sp, sp, #32
; CHECK-IOS-AUTO: stp x29, x30, [sp, #16]
; ...
; CHECK-IOS-AUTO: ldp x29, x30, [sp, #16]
; CHECK-IOS-AUTO: and x29, x29, #0xefffffffffffffff
; CHECK-IOS-AUTO: add sp, sp, #32

; CHECK-WATCHOS-NEVER: sub sp, sp, #32
; CHECK-WATCHOS-NEVER: stp x29, x30, [sp, #16]
; ...
; CHECK-WATCHOS-NEVER: ldp x29, x30, [sp, #16]
; CHECK-WATCHOS-NEVER: add sp, sp, #32

; CHECK-WATCHOS-ALWAYS: orr x29, x29, #0x1000000000000000
; CHECK-WATCHOS-ALWAYS: sub sp, sp, #32
; CHECK-WATCHOS-ALWAYS: stp x29, x30, [sp, #16]
; ...
; CHECK-WATCHOS-ALWAYS: ldp x29, x30, [sp, #16]
; CHECK-WATCHOS-ALWAYS: and x29, x29, #0xefffffffffffffff
; CHECK-WATCHOS-ALWAYS: add sp, sp, #32

; CHECK-WATCHOS-AUTO: adrp x16, _swift_async_extendedFramePointerFlags@GOTPAGE
; CHECK-WATCHOS-AUTO: ldr w16, [x16, _swift_async_extendedFramePointerFlags@GOTPAGEOFF]
; CHECK-WATCHOS-AUTO: orr x29, x29, x16, lsl #32
; CHECK-WATCHOS-AUTO: sub sp, sp, #32
; CHECK-WATCHOS-AUTO: stp x29, x30, [sp, #16]
; ...
; CHECK-WATCHOS-AUTO: ldp x29, x30, [sp, #16]
; CHECK-WATCHOS-AUTO: and x29, x29, #0xefffffffffffffff
; CHECK-WATCHOS-AUTO: add sp, sp, #32
