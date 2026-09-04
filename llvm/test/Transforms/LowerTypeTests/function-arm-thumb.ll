; REQUIRES: arm-registered-target

; RUN: rm -rf %t && split-file %s %t
; RUN: opt -S -mtriple=arm-unknown-linux-gnu -passes=lowertypetests -lowertypetests-summary-action=export \
; RUN:   -lowertypetests-read-summary=%t/summary.ll -lowertypetests-write-summary=%t/out.summary %t/main.ll | FileCheck %s

;--- main.ll
target datalayout = "e-p:64:64"

define void @f1() "target-features"="+thumb-mode,+v6t2" !type !0 {
  ret void
}

define void @g1() "target-features"="-thumb-mode" !type !0 {
  ret void
}

define void @f2() "target-features"="+thumb-mode" !type !1 {
  ret void
}

define void @g2() "target-features"="-thumb-mode" !type !1 {
  ret void
}

define void @h2() "target-features"="-thumb-mode" !type !1 {
  ret void
}

declare void @takeaddr(ptr, ptr, ptr, ptr, ptr)
define void @addrtaken() {
  call void @takeaddr(ptr @f1, ptr @g1, ptr @f2, ptr @g2, ptr @h2)
  ret void
}

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}

; CHECK: define private void {{.*}} #[[AT:.*]] prefalign(4)
; CHECK-NEXT: entry:
; CHECK-NEXT:  call void asm sideeffect "b.w $0\0A", "s"(ptr @f1.cfi)
; CHECK-NEXT:  call void asm sideeffect "b.w $0\0A", "s"(ptr @g1.cfi)
; CHECK-NEXT:  unreachable
; CHECK-NEXT: }

; CHECK: define private void {{.*}} #[[AA:.*]] prefalign(4)
; CHECK-NEXT: entry:
; CHECK-NEXT:  call void asm sideeffect "b $0\0A", "s"(ptr @f2.cfi)
; CHECK-NEXT:  call void asm sideeffect "b $0\0A", "s"(ptr @g2.cfi)
; CHECK-NEXT:  call void asm sideeffect "b $0\0A", "s"(ptr @h2.cfi)
; CHECK-NEXT:  unreachable
; CHECK-NEXT: }

; CHECK-DAG: attributes #[[AA]] = { naked noinline "target-features"="-thumb-mode" }
; CHECK-DAG: attributes #[[AT]] = { naked noinline "target-cpu"="cortex-a8" "target-features"="+thumb-mode" }

;--- summary.ll
^0 = module: (path: "use.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 42, summaries: (function: (module: ^0, flags: (live: 1), insts: 1, typeIdInfo: (typeTests: (14276520915468743435, 15427464259790519041)))))
