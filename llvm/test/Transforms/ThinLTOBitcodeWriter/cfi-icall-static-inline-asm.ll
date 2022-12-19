; REQUIRES: x86-registered-target
; RUN: opt  -thinlto-bc -thinlto-split-lto-unit -o - %s | llvm-modextract -b -n 0 -o - | llvm-dis  | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: module asm ".lto_set_conditional a,a.[[HASH:[0-9a-f]+]]"

define void @b() {
  %f = alloca ptr, align 8
  ; CHECK: store{{.*}} @a.[[HASH]],{{.*}} %f
  store ptr @a, ptr %f, align 8
  ; CHECK: %1 = call ptr asm sideeffect "leaq a(%rip)
  %1 = call ptr asm sideeffect "leaq a(%rip), $0\0A\09", "=r,~{dirflag},~{fpsr},~{flags}"()
  ret void
}

; CHECK: define{{.*}} @a.[[HASH]](){{.*}} !type
define internal void @a() !type !0 {
  ret void
}

!0 = !{i64 0, !"typeid1"}
