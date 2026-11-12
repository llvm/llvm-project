; RUN: rm -rf %t && split-file %s %t
; RUN: opt -S -mtriple=x86_64-unknown-linux -passes=lowertypetests %t/profile.ll | FileCheck --check-prefix=HOT %s
; RUN: opt -S -mtriple=x86_64-unknown-linux -passes=lowertypetests %t/csprofile.ll | FileCheck --check-prefix=HOT %s
; RUN: opt -S -mtriple=x86_64-unknown-linux -passes=lowertypetests %t/noprofile.ll | FileCheck --check-prefix=NOHOT %s

; HOT: define private void @.cfi.jumptable() #[[ATTR:[0-9]+]] prefalign(8) !section_prefix ![[PREFIX:[0-9]+]] !elf_section_properties
; HOT: attributes #[[ATTR]] = { hot naked nocf_check noinline }
; HOT: ![[PREFIX]] = !{!"section_prefix", !"hot"}

; NOHOT: define private void @.cfi.jumptable() #[[ATTR:[0-9]+]] prefalign(8) !elf_section_properties
; NOHOT-NOT: !section_prefix
; NOHOT: attributes #[[ATTR]] = { naked nocf_check noinline }

;--- profile.ll
target datalayout = "e-p:64:64"

@0 = private unnamed_addr constant [2 x ptr] [ptr @f, ptr @g], align 16

define void @f() !type !0 {
  ret void
}

define void @g() !type !0 {
  ret void
}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset)

define i1 @foo(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

!0 = !{i32 0, !"typeid1"}
!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !{}}

;--- csprofile.ll
target datalayout = "e-p:64:64"

@0 = private unnamed_addr constant [2 x ptr] [ptr @f, ptr @g], align 16

define void @f() !type !0 {
  ret void
}

define void @g() !type !0 {
  ret void
}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset)

define i1 @foo(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

!0 = !{i32 0, !"typeid1"}
!llvm.module.flags = !{!1}
!1 = !{i32 1, !"CSProfileSummary", !{}}

;--- noprofile.ll
target datalayout = "e-p:64:64"

@0 = private unnamed_addr constant [2 x ptr] [ptr @f, ptr @g], align 16

define void @f() !type !0 {
  ret void
}

define void @g() !type !0 {
  ret void
}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset)

define i1 @foo(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

!0 = !{i32 0, !"typeid1"}
