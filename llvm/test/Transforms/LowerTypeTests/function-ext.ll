; RUN: opt -S -lowertypetests -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck --check-prefix=X64 %s
; RUN: opt -S -lowertypetests -mtriple=wasm32-unknown-unknown < %s | FileCheck --check-prefix=WASM32 %s

; Tests that we correctly handle external references, including the case where
; all functions in a bitset are external references.

; WASM32: private constant [0 x i8] zeroinitializer

; WASM32: declare !type !{{[0-9]+}} void @foo()
declare !type !0 void @foo()

define i1 @bar(i8* %ptr) {
  ; X64: icmp eq i64 {{.*}}, ptrtoint (void ()* @[[JT:.*]] to i64)
  ; WASM32: ret i1 false
  %p = call i1 @llvm.type.test(i8* %ptr, metadata !"void")
  ret i1 %p
}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

!0 = !{i64 0, !"void"}
; WASM-NOT: !{i64 0}
; WASM-NOT: !{i64 1}

; X64: define private void @[[JT]]() #{{.*}} align {{.*}} {
; X64:   call void asm sideeffect "jmp ${0:c}@plt\0Aint3\0Aint3\0Aint3\0A", "s"(void ()* @foo)
