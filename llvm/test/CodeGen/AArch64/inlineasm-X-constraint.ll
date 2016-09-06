; RUN: llc -mtriple=aarch64-none-linux-gnu < %s -o - | FileCheck %s

; The following functions test the use case where an X constraint is used to
; add a dependency between an assembly instruction (vmsr in this case) and
; another instruction. In each function, we use a different type for the
; X constraint argument.
;
; We can something similar from the following C code:
; double f1(double f, int pscr_value) {
;   asm volatile("msr fpsr,%1" : "=X" ((f)): "r" (pscr_value));
;   return f+f;
; }

; CHECK-LABEL: f1
; CHECK: msr FPSR
; CHECK: fadd d

define  double @f1(double %f, i32 %pscr_value) {
entry:
  %f.addr = alloca double, align 8
  store double %f, double* %f.addr, align 8
  call void asm sideeffect "msr fpsr,$1", "=*X,r"(double* nonnull %f.addr, i32 %pscr_value) nounwind
  %0 = load double, double* %f.addr, align 8
  %add = fadd double %0, %0
  ret double %add
}

; int f2(int f, int pscr_value) {
;   asm volatile("msr fpsr,$1" : "=X" ((f)): "r" (pscr_value));
;   return f*f;
; }

; CHECK-LABEL: f2
; CHECK: msr FPSR
; CHECK: mul
define  i32 @f2(i32 %f, i32 %pscr_value) {
entry:
  %f.addr = alloca i32, align 4
  store i32 %f, i32* %f.addr, align 4
  call void asm sideeffect "msr fpsr,$1", "=*X,r"(i32* nonnull %f.addr, i32 %pscr_value) nounwind
  %0 = load i32, i32* %f.addr, align 4
  %mul = mul i32 %0, %0
  ret i32 %mul
}

; typedef signed char int8_t;
; typedef __attribute__((neon_vector_type(8))) int8_t int8x8_t;
; void f3 (void)
; {
;   int8x8_t vector_res_int8x8;
;   unsigned int fpscr;
;   asm volatile ("msr fpsr,$1" : "=X" ((vector_res_int8x8)) : "r" (fpscr));
;   return vector_res_int8x8 * vector_res_int8x8;
; }

; CHECK-LABEL: f3
; CHECK: msr FPSR
; CHECK: mul
define  <8 x i8> @f3() {
entry:
  %vector_res_int8x8 = alloca <8 x i8>, align 8
  %0 = getelementptr inbounds <8 x i8>, <8 x i8>* %vector_res_int8x8, i32 0, i32 0
  call void asm sideeffect "msr fpsr,$1", "=*X,r"(<8 x i8>* nonnull %vector_res_int8x8, i32 undef) nounwind
  %1 = load <8 x i8>, <8 x i8>* %vector_res_int8x8, align 8
  %mul = mul <8 x i8> %1, %1
  ret <8 x i8> %mul
}

; We can emit integer constants.
; We can get this from:
; void f() {
;   int x = 2;
;   asm volatile ("add x0, x0, %0" : : "X" (x));
; }
;
; CHECK-LABEL: f4
; CHECK: add x0, x0, #2
define void @f4() {
entry:
  tail call void asm sideeffect "add x0, x0, $0", "X"(i32 2)
  ret void
}

; We can emit function labels. This is equivalent to the following C code:
; void f(void) {
;   void (*x)(void) = &foo;
;   asm volatile ("bl %0" : : "X" (x));
; }
; CHECK-LABEL: f5
; CHECK: bl f4
define void @f5() {
entry:
  tail call void asm sideeffect "bl $0", "X"(void ()* nonnull @f4)
  ret void
}

declare void @foo(...)

; This tests the behavior of the X constraint when used on functions pointers,
; or functions with a cast. In the first asm call we figure out that this
; is a function pointer and emit the label. However, in the second asm call
; we can't see through the bitcast and we end up having to lower this constraint
; to something else. This is not ideal, but it is a correct behaviour according
; to the definition of the X constraint.
;
; In this case (and other cases where we could have emitted something else),
; what we're doing with the X constraint is not particularly useful either,
; since the user could have used "r" in this situation for the same effect.

; CHECK-LABEL: f6
; CHECK: bl foo
; CHECK: br x

define void @f6() nounwind {
entry:
  tail call void asm sideeffect "bl $0", "X"(void (...)* @foo) nounwind
  tail call void asm sideeffect "br $0", "X"(void (...)* bitcast (void ()* @f4 to void (...)*)) nounwind
  ret void
}

; The following IR can be generated from C code with a function like:
; void a() {
;   void* a = &&A;
;   asm volatile ("bl %0" : : "X" (a));
;  A:
;   return;
; }
;
; Ideally this would give the block address of bb, but it requires us to see
; through blockaddress, which we can't do at the moment. This might break some
; existing use cases where a user would expect to get a block label and instead
; gets the block address in a register. However, note that according to the
; "no constraints" definition this behaviour is correct (although not very nice).

; CHECK-LABEL: f7
; CHECK: bl
define void @f7() {
  call void asm sideeffect "br $0", "X"( i8* blockaddress(@f7, %bb) )
  br label %bb
bb:
  ret void
}

; If we use a constraint "=*X", we should get a store back to *%x (in x0).
; CHECK-LABEL: f8
; CHECK: add	 [[Dest:x[0-9]+]], x0, x0
; CHECK: str	[[Dest]], [x0]
define void @f8(i64 *%x) {
entry:
  tail call void asm sideeffect "add $0, x0, x0", "=*X"(i64 *%x)
  ret void
}
