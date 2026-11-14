; RUN: not opt -passes='require<libcall-lowering-info>,safe-stack' -mtriple=x86_64-pc-linux-gnu -S %s 2> %t.err | FileCheck %s
; RUN: FileCheck --check-prefix=ERR --implicit-check-not=error %s < %t.err

; A function named __safestack_unsafe_stack_ptr already exists.
declare void @__safestack_unsafe_stack_ptr()

; ERR: error: __safestack_unsafe_stack_ptr must be a global variable

; CHECK-LABEL: define void @foo(
; CHECK: [[USP:%.*]] = load ptr, ptr @__safestack_unsafe_stack_ptr
; CHECK: store ptr {{%.*}}, ptr @__safestack_unsafe_stack_ptr
; CHECK: store ptr [[USP]], ptr @__safestack_unsafe_stack_ptr
define void @foo() nounwind uwtable safestack {
  %a = alloca i8, align 8
  call void @Capture(ptr %a)
  ret void
}

declare void @Capture(ptr)
