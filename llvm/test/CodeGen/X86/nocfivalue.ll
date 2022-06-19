; RUN: opt -S -lowertypetests %s | llc -asm-verbose=false | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: a:
; CHECK-NEXT: .quad   f1
; CHECK-NEXT: .quad   .L.cfi.jumptable
; CHECK-NEXT: .quad   .L.cfi.jumptable
; CHECK-NEXT: .quad   f2
; CHECK-NEXT: .quad   f3
; CHECK-NEXT: .quad   f3.cfi
@a = global [6 x void ()*] [void ()* no_cfi @f1, void ()* @f1, void ()* @f2, void ()* no_cfi @f2, void ()* @f3, void ()* no_cfi @f3]

declare !type !0 void @f1()

define internal void @f2() !type !0 {
  ret void
}

define void @f3() #0 !type !0 {
  ret void
}

declare i1 @llvm.type.test(i8* %ptr, metadata %bitset) nounwind readnone

define i1 @foo(i8* %p) {
  %x = call i1 @llvm.type.test(i8* %p, metadata !"typeid1")
  ret i1 %x
}

!llvm.module.flags = !{!1}

attributes #0 = { "cfi-canonical-jump-table" }

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
