;; Test that symbols defined in shared libraries prevent devirtualization.

;; First check that we get devirtualization when the defs are in the
;; LTO unit.

;; Index based WPD
;; Generate unsplit module with summary for ThinLTO index-based WPD.
; RUN: opt --thinlto-bc -o %t1a.o %s
; RUN: opt --thinlto-bc -o %t2a.o %S/Inputs/devirt_vcall_vis_shared_def.ll
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:   --plugin-opt=whole-program-visibility \
; RUN:   --plugin-opt=save-temps \
; RUN:   --plugin-opt=-pass-remarks=. \
; RUN:   %t1a.o %t2a.o -o %t3a 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1a.o.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

;; Hybrid WPD
;; Generate split module with summary for hybrid Thin/Regular LTO WPD.
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t1b.o %s
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t2b.o %S/Inputs/devirt_vcall_vis_shared_def.ll
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:   --plugin-opt=whole-program-visibility \
; RUN:   --plugin-opt=save-temps \
; RUN:   --plugin-opt=-pass-remarks=. \
; RUN:   %t1b.o %t2b.o -o %t3b 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t1b.o.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

;; Regular LTO WPD
; RUN: opt -o %t1c.o %s
; RUN: opt -o %t2c.o %S/Inputs/devirt_vcall_vis_shared_def.ll
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:   --plugin-opt=whole-program-visibility \
; RUN:   --plugin-opt=save-temps \
; RUN:   --plugin-opt=-pass-remarks=. \
; RUN:   %t1c.o %t2c.o -o %t3c 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: llvm-dis %t3c.0.4.opt.bc -o - | FileCheck %s --check-prefix=CHECK-IR

; REMARK-DAG: single-impl: devirtualized a call to _ZN1A1nEi

;; Check that WPD fails with when linking against a shared library
;; containing the strong defs of the vtables.
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:   %t2c.o -o %t.so -shared

;; Index based WPD
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:   --plugin-opt=whole-program-visibility \
; RUN:   --plugin-opt=-pass-remarks=. \
; RUN:   %t1a.o %t.so -o %t4a \
; RUN:   --export-dynamic 2>&1 | count 0

;; Hybrid WPD
; RUN: opt --thinlto-bc --thinlto-split-lto-unit -o %t4.o %s
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:   --plugin-opt=whole-program-visibility \
; RUN:   --plugin-opt=-pass-remarks=. \
; RUN:   %t1b.o %t.so -o %t4b \
; RUN:   --export-dynamic 2>&1 | count 0

;; Regular LTO WPD
; RUN: opt -o %t4.o %s
; RUN: %gold -m elf_x86_64 -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:   --plugin-opt=whole-program-visibility \
; RUN:   --plugin-opt=-pass-remarks=. \
; RUN:   %t1c.o %t.so -o %t4c \
; RUN:   --export-dynamic 2>&1 | count 0

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

%struct.A = type { ptr }
%struct.B = type { %struct.A }

@_ZTV1A = available_externally unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1A1fEi, ptr @_ZN1A1nEi] }, !type !0, !vcall_visibility !2
@_ZTV1B = linkonce_odr unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr undef, ptr @_ZN1B1fEi, ptr @_ZN1A1nEi] }, !type !0, !type !1, !vcall_visibility !2

;; Prevent the vtables from being dead code eliminated.
@llvm.used = appending global [2 x ptr] [ ptr @_ZTV1A, ptr @_ZTV1B]

; CHECK-IR-LABEL: define dso_local i32 @_start
define i32 @_start(ptr %obj, i32 %a) {
entry:
  %vtable = load ptr, ptr %obj
  %p = call i1 @llvm.type.test(ptr %vtable, metadata !"_ZTS1A")
  call void @llvm.assume(i1 %p)
  %fptrptr = getelementptr ptr, ptr %vtable, i32 1
  %fptr1 = load ptr, ptr %fptrptr, align 8

  ;; Check that the call was devirtualized.
  ; CHECK-IR: %call = tail call i32 @_ZN1A1nEi
  ; CHECK-NODEVIRT-IR: %call = tail call i32 %fptr1
  %call = tail call i32 %fptr1(ptr nonnull %obj, i32 %a)

  ret i32 %call
}
; CHECK-IR-LABEL: ret i32
; CHECK-IR-LABEL: }

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

define available_externally i32 @_ZN1A1fEi(ptr %this, i32 %a) #0 {
   ret i32 0
}

define available_externally i32 @_ZN1A1nEi(ptr %this, i32 %a) #0 {
   ret i32 0
}

define linkonce_odr i32 @_ZN1B1fEi(ptr %this, i32 %a) #0 {
   ret i32 0
}

;; Make sure we don't inline or otherwise optimize out the direct calls.
attributes #0 = { noinline optnone }

!0 = !{i64 16, !"_ZTS1A"}
!1 = !{i64 16, !"_ZTS1B"}
!2 = !{i64 0}
