; RUN: split-file %s %t
; RUN: opt -passes='require<libcall-lowering-info>,safe-stack' -S %t/new-decl.ll | FileCheck --check-prefix=NEW %s
; RUN: opt -passes='require<libcall-lowering-info>,safe-stack' -S %t/existing-decl.ll | FileCheck --check-prefix=DECL %s
; RUN: opt -passes='require<libcall-lowering-info>,safe-stack' -S %t/alias.ll | FileCheck --check-prefix=ALIAS %s
; RUN: opt -passes='require<libcall-lowering-info>,safe-stack' -S %t/variable.ll | FileCheck --check-prefix=VAR %s

; Existing globals with the names of the libcalls are used as-is.

;--- new-decl.ll
; NEW: define void @foo()
; NEW: call arm_aapcscc ptr @__safestack_pointer_address()
; NEW: call arm_aapcscc void @__stack_chk_fail()
; NEW: declare arm_aapcscc ptr @__safestack_pointer_address()
; NEW: declare arm_aapcscc void @__stack_chk_fail()

target triple = "armv7-unknown-linux-android"

define void @foo() safestack sspreq {
  %a = alloca i8, align 8
  call void @Capture(ptr %a)
  ret void
}

declare void @Capture(ptr)

;--- existing-decl.ll
; DECL: declare void @__stack_chk_fail()
; DECL: declare ptr @__safestack_pointer_address()
; DECL: define void @foo()
; DECL: call arm_aapcscc ptr @__safestack_pointer_address()
; DECL: call arm_aapcscc void @__stack_chk_fail()

target triple = "armv7-unknown-linux-android"

declare void @__stack_chk_fail()
declare ptr @__safestack_pointer_address()

define void @foo() safestack sspreq {
  %a = alloca i8, align 8
  call void @Capture(ptr %a)
  ret void
}

declare void @Capture(ptr)

;--- alias.ll
; ALIAS: @__stack_chk_fail = hidden alias void (), ptr @__stack_chk_fail_impl
; ALIAS: @__safestack_pointer_address = hidden alias ptr (), ptr @__safestack_pointer_address_impl
; ALIAS: define void @foo()
; ALIAS: call arm_aapcscc ptr @__safestack_pointer_address()
; ALIAS: call arm_aapcscc void @__stack_chk_fail()

target triple = "armv7-unknown-linux-android"

define hidden void @__stack_chk_fail_impl() {
  unreachable
}

@__stack_chk_fail = hidden alias void (), ptr @__stack_chk_fail_impl

define hidden ptr @__safestack_pointer_address_impl() {
  ret ptr null
}

@__safestack_pointer_address = hidden alias ptr (), ptr @__safestack_pointer_address_impl

define void @foo() safestack sspreq {
  %a = alloca i8, align 8
  call void @Capture(ptr %a)
  ret void
}

declare void @Capture(ptr)

;--- variable.ll
; VAR: @__stack_chk_fail = global ptr null
; VAR: @__safestack_pointer_address = global ptr null
; VAR: define void @foo()
; VAR: call arm_aapcscc ptr @__safestack_pointer_address()
; VAR: call arm_aapcscc void @__stack_chk_fail()

target triple = "armv7-unknown-linux-android"

@__stack_chk_fail = global ptr null
@__safestack_pointer_address = global ptr null

define void @foo() safestack sspreq {
  %a = alloca i8, align 8
  call void @Capture(ptr %a)
  ret void
}

declare void @Capture(ptr)
