; RUN: opt -passes='cgscc(function-attrs),rpo-function-attrs,attributor' \
; RUN:     -attributor-assume-closed-world -S < %s | FileCheck %s

; Nothing in this module marks @dispatcher norecurse. rpo-function-attrs adds it:
; addNoRecurseAttrsTopDown only walks a function's uses, never its body, so a
; function whose body is an indirect call qualifies as long as every caller is
; norecurse. The Attributor then replaces that indirect call with a direct call
; to @outer, and @outer already calls @dispatcher, so @dispatcher is recursive
; and the attribute is false.
;
; TargetFrameLowering::isSafeForNoCSROpt requires norecurse before a function may
; skip saving callee-saved registers, and AMDGPUResourceUsageAnalysis uses it to
; decide a call graph has no recursion.

@g = external global i32

define internal void @inner() norecurse {
  store i32 1, ptr @g
  ret void
}

define internal void @outer() norecurse {
  call void @dispatcher(ptr @inner)
  ret void
}

define internal void @dispatcher(ptr %fn) {
  call void %fn()
  ret void
}

define void @entry() norecurse {
  call void @dispatcher(ptr @outer)
  ret void
}

; The cycle @dispatcher -> @outer -> @dispatcher exists after specialization.
; CHECK: define internal void @outer() #[[ATTR:[0-9]+]] {
; CHECK:   call void @dispatcher(
; CHECK: define internal void @dispatcher(ptr {{.*}}) #[[ATTR]] {
; CHECK:   call void @outer()

; So neither of them may keep norecurse.
; CHECK: attributes #[[ATTR]] = { mustprogress nofree nosync nounwind willreturn memory(write) }
