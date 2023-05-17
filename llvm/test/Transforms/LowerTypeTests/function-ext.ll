; RUN: opt -S -passes=lowertypetests -mtriple=x86_64-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,X64 %s
; RUN: opt -S -passes=lowertypetests -mtriple=wasm32-unknown-unknown %s | FileCheck --check-prefixes=CHECK,WASM32 %s

; Tests that we correctly handle external references, including the case where
; all functions in a bitset are external references.

; WASM32: private constant [0 x i8] zeroinitializer

; WASM32: declare !type !{{[0-9]+}} !wasm.index !{{[0-9]+}} void @foo1()
declare !type !0 void @foo1()
; WASM32: declare !type !{{[0-9]+}} void @foo2()
declare !type !1 void @foo2()

; CHECK-LABEL: @bar
define i1 @bar(ptr %ptr) {
  ; CHECK: %[[ICMP:[0-9]+]] = icmp eq
  ; CHECK: ret i1 %[[ICMP]]
  %p = call i1 @llvm.type.test(ptr %ptr, metadata !"type1")
  ret i1 %p
}

; CHECK-LABEL: @baz
define i1 @baz(ptr %ptr) {
  ; CHECK: ret i1 false
  %p = call i1 @llvm.type.test(ptr %ptr, metadata !"type2")
  ret i1 %p
}

; CHECK-LABEL: @addrtaken
define ptr @addrtaken() {
  ; X64: ret ptr @[[JT:.*]]
  ret ptr @foo1
}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

!0 = !{i64 0, !"type1"}
!1 = !{i64 0, !"type2"}

; X64: define private void @[[JT]]() #{{.*}} align {{.*}} {
; X64:   call void asm sideeffect "jmp ${0:c}@plt\0Aint3\0Aint3\0Aint3\0A", "s"(ptr @foo1)
