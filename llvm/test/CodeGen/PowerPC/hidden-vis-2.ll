; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | FileCheck %s

; CHECK: lis 3, x@ha
; CHECK: lis 4, y@ha
; CHECK: lwz 3, x@l(3)
; CHECK: lwz 4, y@l(4)
; CHECK: .hidden x
; CHECK: .hidden y

@x = external hidden global i32
@y = extern_weak hidden global i32

define i32 @t() nounwind readonly {
entry:
        %0 = load i32, ptr @x, align 4
        %1 = load i32, ptr @y, align 4
        %2 = add i32 %1, %0
        ret i32 %2
}
