; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin9 | FileCheck %s

; CHECK: lis r2, ha16(L_x$non_lazy_ptr)
; CHECK: lis r3, ha16(L_y$non_lazy_ptr)
; CHECK: lwz r2, lo16(L_x$non_lazy_ptr)(r2)
; CHECK: lwz r3, lo16(L_y$non_lazy_ptr)(r3)
; CHECK: L_x$non_lazy_ptr:
; CHECK: L_y$non_lazy_ptr:

@x = external hidden global i32
@y = extern_weak hidden global i32

define i32 @t() nounwind readonly {
entry:
        %0 = load i32, i32* @x, align 4
        %1 = load i32, i32* @y, align 4
        %2 = add i32 %1, %0
        ret i32 %2
}
