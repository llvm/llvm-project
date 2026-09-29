; RUN: llc -mtriple=x86_64-unknown-linux-gnu -stop-after=finalize-isel %s -o - | FileCheck %s

@i = thread_local global i32 0, align 4

define noundef i32 @foo() {
; CHECK: %0:gr64 = MOV64rm $rip, 1, $noreg, target-flags(x86-gottpoff) @i, $noreg :: (load (s64) from got)
; CHECK: %1:gr32 = MOV32rm %0, 1, $noreg, 0, $fs :: (dereferenceable load (s32) from %ir.0)
; CHECK: %2:gr32 = nsw INC32r %1, implicit-def dead $eflags
; CHECK: MOV32mr %0, 1, $noreg, 0, $fs, %2 :: (store (s32) into %ir.0)
; CHECK: $eax = COPY %2
; CHECK: RET 0, $eax
entry:
  %0 = call ptr @llvm.threadlocal.address(ptr @i)
  %1 = load i32, ptr %0, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %0, align 4
  %2 = call ptr @llvm.threadlocal.address(ptr @i)
  %3 = load i32, ptr %2, align 4
  ret i32 %3
}

@j =  thread_local addrspace(1) global  ptr addrspace(0) @i, align 4
define noundef i32 @bar() {
; CHECK: %0:gr64 = MOV64rm $rip, 1, $noreg, target-flags(x86-gottpoff) @j, $noreg :: (load (s64) from got)
; CHECK: %1:gr32 = MOV32rm %0, 1, $noreg, 0, $fs :: (dereferenceable load (s32) from %ir.0, addrspace 1)
; CHECK: %2:gr32 = nsw INC32r %1, implicit-def dead $eflags
; CHECK: MOV32mr %0, 1, $noreg, 0, $fs, %2 :: (store (s32) into %ir.0, addrspace 1)
; CHECK: $eax = COPY %2
; CHECK: RET 0, $eax
entry:
  %0 = call ptr addrspace(1) @llvm.threadlocal.address.p1(ptr addrspace(1) @j)
  %1 = load i32, ptr addrspace(1) %0, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr addrspace(1) %0, align 4
  %2 = call ptr addrspace(1) @llvm.threadlocal.address.p1(ptr addrspace(1) @j)
  %3 = load i32, ptr addrspace(1) %2, align 4
  ret i32 %3
}

declare nonnull ptr @llvm.threadlocal.address(ptr nonnull) nounwind readnone willreturn
declare nonnull ptr addrspace(1) @llvm.threadlocal.address.p1(ptr addrspace(1) nonnull) nounwind readnone willreturn
