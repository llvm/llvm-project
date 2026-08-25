; RUN: opt -passes='cgscc(function-attrs),rpo-function-attrs,attributor' \
; RUN:     -attributor-assume-closed-world -S < %s | FileCheck %s

; rpo-function-attrs is what marks @dispatcher norecurse here:
; addNoRecurseAttrsTopDown only walks a function's uses, never its body, so a
; function whose body is an indirect call qualifies as long as every caller is
; norecurse.

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
