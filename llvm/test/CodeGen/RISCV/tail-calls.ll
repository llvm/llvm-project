; RUN: llc -mtriple riscv32-unknown-linux-gnu -o - %s | FileCheck %s
; RUN: llc -mtriple riscv32-unknown-elf       -o - %s | FileCheck %s

; Perform tail call optimization for global address.
declare i32 @callee_tail(i32 %i)
define i32 @caller_tail(i32 %i) {
; CHECK-LABEL: caller_tail
; CHECK: tail callee_tail
entry:
  %r = tail call i32 @callee_tail(i32 %i)
  ret i32 %r
}

; Perform tail call optimization for external symbol.
@dest = global [2 x i8] zeroinitializer
declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1)
define void @caller_extern(i8* %src) optsize {
entry:
; CHECK: caller_extern
; CHECK-NOT: call memcpy
; CHECK: tail memcpy
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @dest, i32 0, i32 0), i8* %src, i32 7, i1 false)
  ret void
}

; Perform indirect tail call optimization (for function pointer call).
declare void @callee_indirect1()
declare void @callee_indirect2()
define void @caller_indirect_tail(i32 %a) {
; CHECK-LABEL: caller_indirect_tail
; CHECK-NOT: call callee_indirect1
; CHECK-NOT: call callee_indirect2
; CHECK-NOT: tail callee_indirect1
; CHECK-NOT: tail callee_indirect2

; CHECK: lui a0, %hi(callee_indirect2)
; CHECK-NEXT: addi a5, a0, %lo(callee_indirect2)
; CHECK-NEXT: jr a5

; CHECK: lui a0, %hi(callee_indirect1)
; CHECK-NEXT: addi a5, a0, %lo(callee_indirect1)
; CHECK-NEXT: jr a5
entry:
  %tobool = icmp eq i32 %a, 0
  %callee = select i1 %tobool, void ()* @callee_indirect1, void ()* @callee_indirect2
  tail call void %callee()
  ret void
}

; Do not tail call optimize functions with varargs.
declare i32 @callee_varargs(i32, ...)
define void @caller_varargs(i32 %a, i32 %b) {
; CHECK-LABEL: caller_varargs
; CHECK-NOT: tail callee_varargs
; CHECK: call callee_varargs
entry:
  %call = tail call i32 (i32, ...) @callee_varargs(i32 %a, i32 %b, i32 %b, i32 %a)
  ret void
}

; Do not tail call optimize if stack is used to pass parameters.
declare i32 @callee_args(i32 %a, i32 %b, i32 %c, i32 %dd, i32 %e, i32 %ff, i32 %g, i32 %h, i32 %i, i32 %j, i32 %k, i32 %l, i32 %m, i32 %n)
define i32 @caller_args(i32 %a, i32 %b, i32 %c, i32 %dd, i32 %e, i32 %ff, i32 %g, i32 %h, i32 %i, i32 %j, i32 %k, i32 %l, i32 %m, i32 %n) {
; CHECK-LABEL: caller_args
; CHECK-NOT: tail callee_args
; CHECK: call callee_args
entry:
  %r = tail call i32 @callee_args(i32 %a, i32 %b, i32 %c, i32 %dd, i32 %e, i32 %ff, i32 %g, i32 %h, i32 %i, i32 %j, i32 %k, i32 %l, i32 %m, i32 %n)
  ret i32 %r
}

; Do not tail call optimize if parameters need to be passed indirectly.
declare i32 @callee_indirect_args(fp128 %a)
define void @caller_indirect_args() {
; CHECK-LABEL: caller_indirect_args
; CHECK-NOT: tail callee_indirect_args
; CHECK: call callee_indirect_args
entry:
  %call = tail call i32 @callee_indirect_args(fp128 0xL00000000000000003FFF000000000000)
  ret void
}

; Externally-defined functions with weak linkage should not be tail-called.
; The behaviour of branch instructions in this situation (as used for tail
; calls) is implementation-defined, so we cannot rely on the linker replacing
; the tail call with a return.
declare extern_weak void @callee_weak()
define void @caller_weak() {
; CHECK-LABEL: caller_weak
; CHECK-NOT: tail callee_weak
; CHECK: call callee_weak
entry:
  tail call void @callee_weak()
  ret void
}

; Exception-handling functions need a special set of instructions to indicate a
; return to the hardware. Tail-calling another function would probably break
; this.
declare void @callee_irq()
define void @caller_irq() #0 {
; CHECK-LABEL: caller_irq
; CHECK-NOT: tail callee_irq
; CHECK: call callee_irq
entry:
  tail call void @callee_irq()
  ret void
}
attributes #0 = { "interrupt" }

; Byval parameters hand the function a pointer directly into the stack area
; we want to reuse during a tail call. Do not tail call optimize functions with
; byval parameters.
declare i32 @callee_byval(i32** byval %a)
define i32 @caller_byval() {
; CHECK-LABEL: caller_byval
; CHECK-NOT: tail callee_byval
; CHECK: call callee_byval
entry:
  %a = alloca i32*
  %r = tail call i32 @callee_byval(i32** byval %a)
  ret i32 %r
}

; Do not tail call optimize if callee uses structret semantics.
%struct.A = type { i32 }
@a = global %struct.A zeroinitializer

declare void @callee_struct(%struct.A* sret %a)
define void @caller_nostruct() {
; CHECK-LABEL: caller_nostruct
; CHECK-NOT: tail callee_struct
; CHECK: call callee_struct
entry:
  tail call void @callee_struct(%struct.A* sret @a)
  ret void
}

; Do not tail call optimize if caller uses structret semantics.
declare void @callee_nostruct()
define void @caller_struct(%struct.A* sret %a) {
; CHECK-LABEL: caller_struct
; CHECK-NOT: tail callee_nostruct
; CHECK: call callee_nostruct
entry:
  tail call void @callee_nostruct()
  ret void
}
