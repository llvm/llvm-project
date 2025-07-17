;; This is a minimal reproducer that caused StackProtector to crash with a bad cast when
;; CrossDSOCFI is used. This test just needs to not crash.
; REQUIRES: x86-registered-target
; RUN: opt -mtriple=x86_64-pc-linux-gnu %s -passes=lowertypetests,cross-dso-cfi,stack-protector

define hidden void @__stack_chk_fail() !type !1{
  unreachable
}

define void @store_captures() sspstrong {
entry:
  %a = alloca i32, align 4
  %j = alloca ptr, align 8
  store ptr %a, ptr %j, align 8
  ret void
}

define void @func(ptr %0) {
entry:
  %1 = call i1 @llvm.type.test(ptr %0, metadata !"typeid")
  br i1 %1, label %cont, label %trap

trap:                                             ; preds = %entry
  call void @llvm.trap()
  unreachable

cont:                                             ; preds = %entry
  call void %0()
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"Cross-DSO CFI", i32 1}
!1 = !{i64 0, !"typeid"}
