;; Test jump table generation with Indirect Branch Tracking on x86.
; RUN: opt -S -passes=lowertypetests -mtriple=i686 %s | FileCheck --check-prefixes=X86,X86_32 %s
; RUN: opt -S -passes=lowertypetests -mtriple=x86_64 %s | FileCheck --check-prefixes=X86,X86_64 %s

@0 = private unnamed_addr constant [2 x ptr] [ptr @f, ptr @g], align 16

define void @f() !type !0 {
  ret void
}

define internal void @g() !type !0 {
  ret void
}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

define i1 @foo(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

!llvm.module.flags = !{!1}
!0 = !{i32 0, !"typeid1"}
!1 = !{i32 8, !"cf-protection-branch", i32 1}

; X86:         define private void @.cfi.jumptable() #[[#ATTR:]] align 16 {
; X86-NEXT:    entry:
; X86_32-NEXT:   call void asm sideeffect "endbr32\0Ajmp ${0:c}@plt\0A.balign 16, 0xcc\0Aendbr32\0Ajmp ${1:c}@plt\0A.balign 16, 0xcc\0A", "s,s"(ptr @f.cfi, ptr @g.cfi)
; X86_64-NEXT:   call void asm sideeffect "endbr64\0Ajmp ${0:c}@plt\0A.balign 16, 0xcc\0Aendbr64\0Ajmp ${1:c}@plt\0A.balign 16, 0xcc\0A", "s,s"(ptr @f.cfi, ptr @g.cfi)

; X86_64: attributes #[[#ATTR]] = { naked nocf_check nounwind }
