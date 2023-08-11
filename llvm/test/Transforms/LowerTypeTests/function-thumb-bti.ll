; REQUIRES: arm-registered-target

; RUN: sed "s/ENABLE_BTI/1/" %s | opt -S -passes=lowertypetests -mtriple=thumbv8.1m.main-unknown-linux-gnu | FileCheck %s --check-prefixes=CHECK,BTI
; RUN: sed "s/ENABLE_BTI/0/" %s | opt -S -passes=lowertypetests -mtriple=thumbv8.1m.main-unknown-linux-gnu | FileCheck %s --check-prefixes=CHECK,NOBTI

target datalayout = "e-p:64:64"

@0 = private unnamed_addr constant [2 x ptr] [ptr @f, ptr @g], align 16

define void @f() !type !0 {
  ret void
}

define internal void @g() !type !0 {
  ret void
}

!0 = !{i32 0, !"typeid1"}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

define i1 @foo(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

!llvm.module.flags = !{!1}

!1 = !{i32 4, !"branch-target-enforcement", i32 ENABLE_BTI}

; For BTI, expect jump table offset check to involve a shift right by
; 3 because table entries are 8 bytes long, consisting of a BTI and a
; branch instruction, 4 bytes each. For non-BTI, we shift right by 2,
; because it's just the branch.

; BTI:   lshr i64 {{.*}}, 3
; NOBTI: lshr i64 {{.*}}, 2

; CHECK: define private void @.cfi.jumptable() [[ATTRS:#[0-9]+]]

; And check the actual jump table asm string:

; BTI:   call void asm sideeffect "bti\0Ab.w $0\0Abti\0Ab.w $1\0A", "s,s"(ptr @f.cfi, ptr @g.cfi)
; NOBTI: call void asm sideeffect "b.w $0\0Ab.w $1\0A", "s,s"(ptr @f.cfi, ptr @g.cfi)

; BTI: attributes [[ATTRS]] = { naked nounwind "branch-target-enforcement"="false" "sign-return-address"="none" "target-features"="+thumb-mode,+pacbti" }
; NOBTI: attributes [[ATTRS]] = { naked nounwind "branch-target-enforcement"="false" "sign-return-address"="none" "target-cpu"="cortex-a8" "target-features"="+thumb-mode" }
