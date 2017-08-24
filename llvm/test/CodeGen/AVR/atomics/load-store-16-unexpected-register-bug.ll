; RUN: llc < %s -march=avr | FileCheck %s

; At one point, the 16-vit atomic load/store operations we defined in TableGen
; to use 'PTRREGS', but the pseudo expander would generate LDDW/STDW instructions.
;
; This would sometimes cause codegen to fail because LDDW requires 'PTRDISPREGS', and
; so if we attempted to generate an atomic operation on the X register, it would hit
; an assertion;

%AtomicI16 = type { %UnsafeCell, [0 x i8] }
%UnsafeCell = type { i16, [0 x i8] }

; CHECK-LABEL: foo
define i8 @foo(%AtomicI16*) {
start:

; We should not be generating atomics that use the X register, they will fail when emitting MC.
; CHECK-NOT: X
  %1 = getelementptr inbounds %AtomicI16, %AtomicI16* %0, i16 0, i32 0, i32 0
  %2 = load atomic i16, i16* %1 seq_cst, align 2
  ret i8 0
}

