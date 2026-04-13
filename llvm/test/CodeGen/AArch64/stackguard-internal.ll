; RUN: llc -O3 %s -o - | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linux-gnu"

; Make sure we correctly lower stack guards even if __stack_chk_guard
; is an alias.  (The alias is created by GlobalMerge.)
; CHECK: adrp {{.*}}, __stack_chk_guard
; CHECK: ldr {{.*}}, [{{.*}}, :lo12:__stack_chk_guard]
; CHECK: __stack_chk_guard = .L_MergedGlobals+4

@__stack_chk_guard = internal global [8 x i32] zeroinitializer, align 4
@x = internal global i32 0, align 4

define i32 @b() nounwind sspstrong {
entry:
  %z = alloca [10 x i32], align 4
  %call = call i32 @a(ptr @__stack_chk_guard, ptr nonnull @x, ptr nonnull %z) #3
  ret i32 %call
}
declare i32 @a(ptr, ptr, ptr)
