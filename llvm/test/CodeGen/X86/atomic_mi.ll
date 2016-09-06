; RUN: llc < %s -mtriple=x86_64-unknown-unknown -verify-machineinstrs | FileCheck %s --check-prefix X64
; RUN: llc < %s -mtriple=i686-unknown-unknown -verify-machineinstrs | FileCheck %s --check-prefix X32
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=slow-incdec -verify-machineinstrs | FileCheck %s --check-prefix SLOW_INC

; This file checks that atomic (non-seq_cst) stores of immediate values are
; done in one mov instruction and not 2. More precisely, it makes sure that the
; immediate is not first copied uselessly into a register.

; Similarily, it checks that a binary operation of an immediate with an atomic
; variable that is stored back in that variable is done as a single instruction.
; For example: x.store(42 + x.load(memory_order_acquire), memory_order_release)
; should be just an add instruction, instead of loading x into a register, doing
; an add and storing the result back.
; The binary operations supported are currently add, and, or, xor.
; sub is not supported because they are translated by an addition of the
; negated immediate.
;
; We also check the same patterns:
; - For inc/dec.
; - For register instead of immediate operands.
; - For floating point operations.

; seq_cst stores are left as (lock) xchgl, but we try to check every other
; attribute at least once.

; Please note that these operations do not require the lock prefix: only
; sequentially consistent stores require this kind of protection on X86.
; And even for seq_cst operations, llvm uses the xchg instruction which has
; an implicit lock prefix, so making it explicit is not required.

define void @store_atomic_imm_8(i8* %p) {
; X64-LABEL: store_atomic_imm_8:
; X64: movb
; X64-NOT: movb
; X32-LABEL: store_atomic_imm_8:
; X32: movb
; X32-NOT: movb
  store atomic i8 42, i8* %p release, align 1
  ret void
}

define void @store_atomic_imm_16(i16* %p) {
; X64-LABEL: store_atomic_imm_16:
; X64: movw
; X64-NOT: movw
; X32-LABEL: store_atomic_imm_16:
; X32: movw
; X32-NOT: movw
  store atomic i16 42, i16* %p monotonic, align 2
  ret void
}

define void @store_atomic_imm_32(i32* %p) {
; X64-LABEL: store_atomic_imm_32:
; X64: movl
; X64-NOT: movl
;   On 32 bits, there is an extra movl for each of those functions
;   (probably for alignment reasons).
; X32-LABEL: store_atomic_imm_32:
; X32: movl 4(%esp), %eax
; X32: movl
; X32-NOT: movl
  store atomic i32 42, i32* %p release, align 4
  ret void
}

define void @store_atomic_imm_64(i64* %p) {
; X64-LABEL: store_atomic_imm_64:
; X64: movq
; X64-NOT: movq
;   These are implemented with a CAS loop on 32 bit architectures, and thus
;   cannot be optimized in the same way as the others.
; X32-LABEL: store_atomic_imm_64:
; X32: cmpxchg8b
  store atomic i64 42, i64* %p release, align 8
  ret void
}

; If an immediate is too big to fit in 32 bits, it cannot be store in one mov,
; even on X64, one must use movabsq that can only target a register.
define void @store_atomic_imm_64_big(i64* %p) {
; X64-LABEL: store_atomic_imm_64_big:
; X64: movabsq
; X64: movq
  store atomic i64 100000000000, i64* %p monotonic, align 8
  ret void
}

; It would be incorrect to replace a lock xchgl by a movl
define void @store_atomic_imm_32_seq_cst(i32* %p) {
; X64-LABEL: store_atomic_imm_32_seq_cst:
; X64: xchgl
; X32-LABEL: store_atomic_imm_32_seq_cst:
; X32: xchgl
  store atomic i32 42, i32* %p seq_cst, align 4
  ret void
}

; ----- ADD -----

define void @add_8i(i8* %p) {
; X64-LABEL: add_8i:
; X64-NOT: lock
; X64: addb
; X64-NOT: movb
; X32-LABEL: add_8i:
; X32-NOT: lock
; X32: addb
; X32-NOT: movb
  %1 = load atomic i8, i8* %p seq_cst, align 1
  %2 = add i8 %1, 2
  store atomic i8 %2, i8* %p release, align 1
  ret void
}

define void @add_8r(i8* %p, i8 %v) {
; X64-LABEL: add_8r:
; X64-NOT: lock
; X64: addb
; X64-NOT: movb
; X32-LABEL: add_8r:
; X32-NOT: lock
; X32: addb
; X32-NOT: movb
  %1 = load atomic i8, i8* %p seq_cst, align 1
  %2 = add i8 %1, %v
  store atomic i8 %2, i8* %p release, align 1
  ret void
}

define void @add_16i(i16* %p) {
;   Currently the transformation is not done on 16 bit accesses, as the backend
;   treat 16 bit arithmetic as expensive on X86/X86_64.
; X64-LABEL: add_16i:
; X64-NOT: addw
; X32-LABEL: add_16i:
; X32-NOT: addw
  %1 = load atomic i16, i16* %p acquire, align 2
  %2 = add i16 %1, 2
  store atomic i16 %2, i16* %p release, align 2
  ret void
}

define void @add_16r(i16* %p, i16 %v) {
;   Currently the transformation is not done on 16 bit accesses, as the backend
;   treat 16 bit arithmetic as expensive on X86/X86_64.
; X64-LABEL: add_16r:
; X64-NOT: addw
; X32-LABEL: add_16r:
; X32-NOT: addw [.*], (
  %1 = load atomic i16, i16* %p acquire, align 2
  %2 = add i16 %1, %v
  store atomic i16 %2, i16* %p release, align 2
  ret void
}

define void @add_32i(i32* %p) {
; X64-LABEL: add_32i:
; X64-NOT: lock
; X64: addl
; X64-NOT: movl
; X32-LABEL: add_32i:
; X32-NOT: lock
; X32: addl
; X32-NOT: movl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = add i32 %1, 2
  store atomic i32 %2, i32* %p monotonic, align 4
  ret void
}

define void @add_32r(i32* %p, i32 %v) {
; X64-LABEL: add_32r:
; X64-NOT: lock
; X64: addl
; X64-NOT: movl
; X32-LABEL: add_32r:
; X32-NOT: lock
; X32: addl
; X32-NOT: movl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = add i32 %1, %v
  store atomic i32 %2, i32* %p monotonic, align 4
  ret void
}

; The following is a corner case where the load is added to itself. The pattern
; matching should not fold this. We only test with 32-bit add, but the same
; applies to other sizes and operations.
define void @add_32r_self(i32* %p) {
; X64-LABEL: add_32r_self:
; X64-NOT: lock
; X64: movl (%[[M:[a-z]+]]), %[[R:[a-z]+]]
; X64: addl %[[R]], %[[R]]
; X64: movl %[[R]], (%[[M]])
; X32-LABEL: add_32r_self:
; X32-NOT: lock
; X32: movl (%[[M:[a-z]+]]), %[[R:[a-z]+]]
; X32: addl %[[R]], %[[R]]
; X32: movl %[[R]], (%[[M]])
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = add i32 %1, %1
  store atomic i32 %2, i32* %p monotonic, align 4
  ret void
}

; The following is a corner case where the load's result is returned. The
; optimizer isn't allowed to duplicate the load because it's atomic.
define i32 @add_32r_ret_load(i32* %p, i32 %v) {
; X64-LABEL: add_32r_ret_load:
; X64-NOT: lock
; X64:      movl (%rdi), %eax
; X64-NEXT: addl %eax, %esi
; X64-NEXT: movl %esi, (%rdi)
; X64-NEXT: retq
; X32-LABEL: add_32r_ret_load:
; X32-NOT: lock
; X32:      movl 4(%esp), %[[P:[a-z]+]]
; X32-NEXT: movl (%[[P]]),
; X32-NOT: %[[P]]
; More code here, we just don't want it to load from P.
; X32: movl %{{.*}}, (%[[P]])
; X32-NEXT: retl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = add i32 %1, %v
  store atomic i32 %2, i32* %p monotonic, align 4
  ret i32 %1
}

define void @add_64i(i64* %p) {
; X64-LABEL: add_64i:
; X64-NOT: lock
; X64: addq
; X64-NOT: movq
;   We do not check X86-32 as it cannot do 'addq'.
; X32-LABEL: add_64i:
  %1 = load atomic i64, i64* %p acquire, align 8
  %2 = add i64 %1, 2
  store atomic i64 %2, i64* %p release, align 8
  ret void
}

define void @add_64r(i64* %p, i64 %v) {
; X64-LABEL: add_64r:
; X64-NOT: lock
; X64: addq
; X64-NOT: movq
;   We do not check X86-32 as it cannot do 'addq'.
; X32-LABEL: add_64r:
  %1 = load atomic i64, i64* %p acquire, align 8
  %2 = add i64 %1, %v
  store atomic i64 %2, i64* %p release, align 8
  ret void
}

define void @add_32i_seq_cst(i32* %p) {
; X64-LABEL: add_32i_seq_cst:
; X64: xchgl
; X32-LABEL: add_32i_seq_cst:
; X32: xchgl
  %1 = load atomic i32, i32* %p monotonic, align 4
  %2 = add i32 %1, 2
  store atomic i32 %2, i32* %p seq_cst, align 4
  ret void
}

define void @add_32r_seq_cst(i32* %p, i32 %v) {
; X64-LABEL: add_32r_seq_cst:
; X64: xchgl
; X32-LABEL: add_32r_seq_cst:
; X32: xchgl
  %1 = load atomic i32, i32* %p monotonic, align 4
  %2 = add i32 %1, %v
  store atomic i32 %2, i32* %p seq_cst, align 4
  ret void
}

; ----- AND -----

define void @and_8i(i8* %p) {
; X64-LABEL: and_8i:
; X64-NOT: lock
; X64: andb
; X64-NOT: movb
; X32-LABEL: and_8i:
; X32-NOT: lock
; X32: andb
; X32-NOT: movb
  %1 = load atomic i8, i8* %p monotonic, align 1
  %2 = and i8 %1, 2
  store atomic i8 %2, i8* %p release, align 1
  ret void
}

define void @and_8r(i8* %p, i8 %v) {
; X64-LABEL: and_8r:
; X64-NOT: lock
; X64: andb
; X64-NOT: movb
; X32-LABEL: and_8r:
; X32-NOT: lock
; X32: andb
; X32-NOT: movb
  %1 = load atomic i8, i8* %p monotonic, align 1
  %2 = and i8 %1, %v
  store atomic i8 %2, i8* %p release, align 1
  ret void
}

define void @and_16i(i16* %p) {
;   Currently the transformation is not done on 16 bit accesses, as the backend
;   treat 16 bit arithmetic as expensive on X86/X86_64.
; X64-LABEL: and_16i:
; X64-NOT: andw
; X32-LABEL: and_16i:
; X32-NOT: andw
  %1 = load atomic i16, i16* %p acquire, align 2
  %2 = and i16 %1, 2
  store atomic i16 %2, i16* %p release, align 2
  ret void
}

define void @and_16r(i16* %p, i16 %v) {
;   Currently the transformation is not done on 16 bit accesses, as the backend
;   treat 16 bit arithmetic as expensive on X86/X86_64.
; X64-LABEL: and_16r:
; X64-NOT: andw
; X32-LABEL: and_16r:
; X32-NOT: andw [.*], (
  %1 = load atomic i16, i16* %p acquire, align 2
  %2 = and i16 %1, %v
  store atomic i16 %2, i16* %p release, align 2
  ret void
}

define void @and_32i(i32* %p) {
; X64-LABEL: and_32i:
; X64-NOT: lock
; X64: andl
; X64-NOT: movl
; X32-LABEL: and_32i:
; X32-NOT: lock
; X32: andl
; X32-NOT: movl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = and i32 %1, 2
  store atomic i32 %2, i32* %p release, align 4
  ret void
}

define void @and_32r(i32* %p, i32 %v) {
; X64-LABEL: and_32r:
; X64-NOT: lock
; X64: andl
; X64-NOT: movl
; X32-LABEL: and_32r:
; X32-NOT: lock
; X32: andl
; X32-NOT: movl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = and i32 %1, %v
  store atomic i32 %2, i32* %p release, align 4
  ret void
}

define void @and_64i(i64* %p) {
; X64-LABEL: and_64i:
; X64-NOT: lock
; X64: andq
; X64-NOT: movq
;   We do not check X86-32 as it cannot do 'andq'.
; X32-LABEL: and_64i:
  %1 = load atomic i64, i64* %p acquire, align 8
  %2 = and i64 %1, 2
  store atomic i64 %2, i64* %p release, align 8
  ret void
}

define void @and_64r(i64* %p, i64 %v) {
; X64-LABEL: and_64r:
; X64-NOT: lock
; X64: andq
; X64-NOT: movq
;   We do not check X86-32 as it cannot do 'andq'.
; X32-LABEL: and_64r:
  %1 = load atomic i64, i64* %p acquire, align 8
  %2 = and i64 %1, %v
  store atomic i64 %2, i64* %p release, align 8
  ret void
}

define void @and_32i_seq_cst(i32* %p) {
; X64-LABEL: and_32i_seq_cst:
; X64: xchgl
; X32-LABEL: and_32i_seq_cst:
; X32: xchgl
  %1 = load atomic i32, i32* %p monotonic, align 4
  %2 = and i32 %1, 2
  store atomic i32 %2, i32* %p seq_cst, align 4
  ret void
}

define void @and_32r_seq_cst(i32* %p, i32 %v) {
; X64-LABEL: and_32r_seq_cst:
; X64: xchgl
; X32-LABEL: and_32r_seq_cst:
; X32: xchgl
  %1 = load atomic i32, i32* %p monotonic, align 4
  %2 = and i32 %1, %v
  store atomic i32 %2, i32* %p seq_cst, align 4
  ret void
}

; ----- OR -----

define void @or_8i(i8* %p) {
; X64-LABEL: or_8i:
; X64-NOT: lock
; X64: orb
; X64-NOT: movb
; X32-LABEL: or_8i:
; X32-NOT: lock
; X32: orb
; X32-NOT: movb
  %1 = load atomic i8, i8* %p acquire, align 1
  %2 = or i8 %1, 2
  store atomic i8 %2, i8* %p release, align 1
  ret void
}

define void @or_8r(i8* %p, i8 %v) {
; X64-LABEL: or_8r:
; X64-NOT: lock
; X64: orb
; X64-NOT: movb
; X32-LABEL: or_8r:
; X32-NOT: lock
; X32: orb
; X32-NOT: movb
  %1 = load atomic i8, i8* %p acquire, align 1
  %2 = or i8 %1, %v
  store atomic i8 %2, i8* %p release, align 1
  ret void
}

define void @or_16i(i16* %p) {
; X64-LABEL: or_16i:
; X64-NOT: orw
; X32-LABEL: or_16i:
; X32-NOT: orw
  %1 = load atomic i16, i16* %p acquire, align 2
  %2 = or i16 %1, 2
  store atomic i16 %2, i16* %p release, align 2
  ret void
}

define void @or_16r(i16* %p, i16 %v) {
; X64-LABEL: or_16r:
; X64-NOT: orw
; X32-LABEL: or_16r:
; X32-NOT: orw [.*], (
  %1 = load atomic i16, i16* %p acquire, align 2
  %2 = or i16 %1, %v
  store atomic i16 %2, i16* %p release, align 2
  ret void
}

define void @or_32i(i32* %p) {
; X64-LABEL: or_32i:
; X64-NOT: lock
; X64: orl
; X64-NOT: movl
; X32-LABEL: or_32i:
; X32-NOT: lock
; X32: orl
; X32-NOT: movl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = or i32 %1, 2
  store atomic i32 %2, i32* %p release, align 4
  ret void
}

define void @or_32r(i32* %p, i32 %v) {
; X64-LABEL: or_32r:
; X64-NOT: lock
; X64: orl
; X64-NOT: movl
; X32-LABEL: or_32r:
; X32-NOT: lock
; X32: orl
; X32-NOT: movl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = or i32 %1, %v
  store atomic i32 %2, i32* %p release, align 4
  ret void
}

define void @or_64i(i64* %p) {
; X64-LABEL: or_64i:
; X64-NOT: lock
; X64: orq
; X64-NOT: movq
;   We do not check X86-32 as it cannot do 'orq'.
; X32-LABEL: or_64i:
  %1 = load atomic i64, i64* %p acquire, align 8
  %2 = or i64 %1, 2
  store atomic i64 %2, i64* %p release, align 8
  ret void
}

define void @or_64r(i64* %p, i64 %v) {
; X64-LABEL: or_64r:
; X64-NOT: lock
; X64: orq
; X64-NOT: movq
;   We do not check X86-32 as it cannot do 'orq'.
; X32-LABEL: or_64r:
  %1 = load atomic i64, i64* %p acquire, align 8
  %2 = or i64 %1, %v
  store atomic i64 %2, i64* %p release, align 8
  ret void
}

define void @or_32i_seq_cst(i32* %p) {
; X64-LABEL: or_32i_seq_cst:
; X64: xchgl
; X32-LABEL: or_32i_seq_cst:
; X32: xchgl
  %1 = load atomic i32, i32* %p monotonic, align 4
  %2 = or i32 %1, 2
  store atomic i32 %2, i32* %p seq_cst, align 4
  ret void
}

define void @or_32r_seq_cst(i32* %p, i32 %v) {
; X64-LABEL: or_32r_seq_cst:
; X64: xchgl
; X32-LABEL: or_32r_seq_cst:
; X32: xchgl
  %1 = load atomic i32, i32* %p monotonic, align 4
  %2 = or i32 %1, %v
  store atomic i32 %2, i32* %p seq_cst, align 4
  ret void
}

; ----- XOR -----

define void @xor_8i(i8* %p) {
; X64-LABEL: xor_8i:
; X64-NOT: lock
; X64: xorb
; X64-NOT: movb
; X32-LABEL: xor_8i:
; X32-NOT: lock
; X32: xorb
; X32-NOT: movb
  %1 = load atomic i8, i8* %p acquire, align 1
  %2 = xor i8 %1, 2
  store atomic i8 %2, i8* %p release, align 1
  ret void
}

define void @xor_8r(i8* %p, i8 %v) {
; X64-LABEL: xor_8r:
; X64-NOT: lock
; X64: xorb
; X64-NOT: movb
; X32-LABEL: xor_8r:
; X32-NOT: lock
; X32: xorb
; X32-NOT: movb
  %1 = load atomic i8, i8* %p acquire, align 1
  %2 = xor i8 %1, %v
  store atomic i8 %2, i8* %p release, align 1
  ret void
}

define void @xor_16i(i16* %p) {
; X64-LABEL: xor_16i:
; X64-NOT: xorw
; X32-LABEL: xor_16i:
; X32-NOT: xorw
  %1 = load atomic i16, i16* %p acquire, align 2
  %2 = xor i16 %1, 2
  store atomic i16 %2, i16* %p release, align 2
  ret void
}

define void @xor_16r(i16* %p, i16 %v) {
; X64-LABEL: xor_16r:
; X64-NOT: xorw
; X32-LABEL: xor_16r:
; X32-NOT: xorw [.*], (
  %1 = load atomic i16, i16* %p acquire, align 2
  %2 = xor i16 %1, %v
  store atomic i16 %2, i16* %p release, align 2
  ret void
}

define void @xor_32i(i32* %p) {
; X64-LABEL: xor_32i:
; X64-NOT: lock
; X64: xorl
; X64-NOT: movl
; X32-LABEL: xor_32i:
; X32-NOT: lock
; X32: xorl
; X32-NOT: movl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = xor i32 %1, 2
  store atomic i32 %2, i32* %p release, align 4
  ret void
}

define void @xor_32r(i32* %p, i32 %v) {
; X64-LABEL: xor_32r:
; X64-NOT: lock
; X64: xorl
; X64-NOT: movl
; X32-LABEL: xor_32r:
; X32-NOT: lock
; X32: xorl
; X32-NOT: movl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = xor i32 %1, %v
  store atomic i32 %2, i32* %p release, align 4
  ret void
}

define void @xor_64i(i64* %p) {
; X64-LABEL: xor_64i:
; X64-NOT: lock
; X64: xorq
; X64-NOT: movq
;   We do not check X86-32 as it cannot do 'xorq'.
; X32-LABEL: xor_64i:
  %1 = load atomic i64, i64* %p acquire, align 8
  %2 = xor i64 %1, 2
  store atomic i64 %2, i64* %p release, align 8
  ret void
}

define void @xor_64r(i64* %p, i64 %v) {
; X64-LABEL: xor_64r:
; X64-NOT: lock
; X64: xorq
; X64-NOT: movq
;   We do not check X86-32 as it cannot do 'xorq'.
; X32-LABEL: xor_64r:
  %1 = load atomic i64, i64* %p acquire, align 8
  %2 = xor i64 %1, %v
  store atomic i64 %2, i64* %p release, align 8
  ret void
}

define void @xor_32i_seq_cst(i32* %p) {
; X64-LABEL: xor_32i_seq_cst:
; X64: xchgl
; X32-LABEL: xor_32i_seq_cst:
; X32: xchgl
  %1 = load atomic i32, i32* %p monotonic, align 4
  %2 = xor i32 %1, 2
  store atomic i32 %2, i32* %p seq_cst, align 4
  ret void
}

define void @xor_32r_seq_cst(i32* %p, i32 %v) {
; X64-LABEL: xor_32r_seq_cst:
; X64: xchgl
; X32-LABEL: xor_32r_seq_cst:
; X32: xchgl
  %1 = load atomic i32, i32* %p monotonic, align 4
  %2 = xor i32 %1, %v
  store atomic i32 %2, i32* %p seq_cst, align 4
  ret void
}

; ----- INC -----

define void @inc_8(i8* %p) {
; X64-LABEL: inc_8:
; X64-NOT: lock
; X64: incb
; X64-NOT: movb
; X32-LABEL: inc_8:
; X32-NOT: lock
; X32: incb
; X32-NOT: movb
; SLOW_INC-LABEL: inc_8:
; SLOW_INC-NOT: incb
; SLOW_INC-NOT: movb
  %1 = load atomic i8, i8* %p seq_cst, align 1
  %2 = add i8 %1, 1
  store atomic i8 %2, i8* %p release, align 1
  ret void
}

define void @inc_16(i16* %p) {
;   Currently the transformation is not done on 16 bit accesses, as the backend
;   treat 16 bit arithmetic as expensive on X86/X86_64.
; X64-LABEL: inc_16:
; X64-NOT: incw
; X32-LABEL: inc_16:
; X32-NOT: incw
; SLOW_INC-LABEL: inc_16:
; SLOW_INC-NOT: incw
  %1 = load atomic i16, i16* %p acquire, align 2
  %2 = add i16 %1, 1
  store atomic i16 %2, i16* %p release, align 2
  ret void
}

define void @inc_32(i32* %p) {
; X64-LABEL: inc_32:
; X64-NOT: lock
; X64: incl
; X64-NOT: movl
; X32-LABEL: inc_32:
; X32-NOT: lock
; X32: incl
; X32-NOT: movl
; SLOW_INC-LABEL: inc_32:
; SLOW_INC-NOT: incl
; SLOW_INC-NOT: movl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = add i32 %1, 1
  store atomic i32 %2, i32* %p monotonic, align 4
  ret void
}

define void @inc_64(i64* %p) {
; X64-LABEL: inc_64:
; X64-NOT: lock
; X64: incq
; X64-NOT: movq
;   We do not check X86-32 as it cannot do 'incq'.
; X32-LABEL: inc_64:
; SLOW_INC-LABEL: inc_64:
; SLOW_INC-NOT: incq
; SLOW_INC-NOT: movq
  %1 = load atomic i64, i64* %p acquire, align 8
  %2 = add i64 %1, 1
  store atomic i64 %2, i64* %p release, align 8
  ret void
}

define void @inc_32_seq_cst(i32* %p) {
; X64-LABEL: inc_32_seq_cst:
; X64: xchgl
; X32-LABEL: inc_32_seq_cst:
; X32: xchgl
  %1 = load atomic i32, i32* %p monotonic, align 4
  %2 = add i32 %1, 1
  store atomic i32 %2, i32* %p seq_cst, align 4
  ret void
}

; ----- DEC -----

define void @dec_8(i8* %p) {
; X64-LABEL: dec_8:
; X64-NOT: lock
; X64: decb
; X64-NOT: movb
; X32-LABEL: dec_8:
; X32-NOT: lock
; X32: decb
; X32-NOT: movb
; SLOW_INC-LABEL: dec_8:
; SLOW_INC-NOT: decb
; SLOW_INC-NOT: movb
  %1 = load atomic i8, i8* %p seq_cst, align 1
  %2 = sub i8 %1, 1
  store atomic i8 %2, i8* %p release, align 1
  ret void
}

define void @dec_16(i16* %p) {
;   Currently the transformation is not done on 16 bit accesses, as the backend
;   treat 16 bit arithmetic as expensive on X86/X86_64.
; X64-LABEL: dec_16:
; X64-NOT: decw
; X32-LABEL: dec_16:
; X32-NOT: decw
; SLOW_INC-LABEL: dec_16:
; SLOW_INC-NOT: decw
  %1 = load atomic i16, i16* %p acquire, align 2
  %2 = sub i16 %1, 1
  store atomic i16 %2, i16* %p release, align 2
  ret void
}

define void @dec_32(i32* %p) {
; X64-LABEL: dec_32:
; X64-NOT: lock
; X64: decl
; X64-NOT: movl
; X32-LABEL: dec_32:
; X32-NOT: lock
; X32: decl
; X32-NOT: movl
; SLOW_INC-LABEL: dec_32:
; SLOW_INC-NOT: decl
; SLOW_INC-NOT: movl
  %1 = load atomic i32, i32* %p acquire, align 4
  %2 = sub i32 %1, 1
  store atomic i32 %2, i32* %p monotonic, align 4
  ret void
}

define void @dec_64(i64* %p) {
; X64-LABEL: dec_64:
; X64-NOT: lock
; X64: decq
; X64-NOT: movq
;   We do not check X86-32 as it cannot do 'decq'.
; X32-LABEL: dec_64:
; SLOW_INC-LABEL: dec_64:
; SLOW_INC-NOT: decq
; SLOW_INC-NOT: movq
  %1 = load atomic i64, i64* %p acquire, align 8
  %2 = sub i64 %1, 1
  store atomic i64 %2, i64* %p release, align 8
  ret void
}

define void @dec_32_seq_cst(i32* %p) {
; X64-LABEL: dec_32_seq_cst:
; X64: xchgl
; X32-LABEL: dec_32_seq_cst:
; X32: xchgl
  %1 = load atomic i32, i32* %p monotonic, align 4
  %2 = sub i32 %1, 1
  store atomic i32 %2, i32* %p seq_cst, align 4
  ret void
}

; ----- FADD -----

define void @fadd_32r(float* %loc, float %val) {
; X64-LABEL: fadd_32r:
; X64-NOT: lock
; X64-NOT: mov
; X64: addss (%[[M:[a-z]+]]), %[[XMM:xmm[0-9]+]]
; X64-NEXT: movss %[[XMM]], (%[[M]])
; X32-LABEL: fadd_32r:
; Don't check x86-32.
; LLVM's SSE handling is conservative on x86-32 even without using atomics.
  %floc = bitcast float* %loc to i32*
  %1 = load atomic i32, i32* %floc seq_cst, align 4
  %2 = bitcast i32 %1 to float
  %add = fadd float %2, %val
  %3 = bitcast float %add to i32
  store atomic i32 %3, i32* %floc release, align 4
  ret void
}

define void @fadd_64r(double* %loc, double %val) {
; X64-LABEL: fadd_64r:
; X64-NOT: lock
; X64-NOT: mov
; X64: addsd (%[[M:[a-z]+]]), %[[XMM:xmm[0-9]+]]
; X64-NEXT: movsd %[[XMM]], (%[[M]])
; X32-LABEL: fadd_64r:
; Don't check x86-32 (see comment above).
  %floc = bitcast double* %loc to i64*
  %1 = load atomic i64, i64* %floc seq_cst, align 8
  %2 = bitcast i64 %1 to double
  %add = fadd double %2, %val
  %3 = bitcast double %add to i64
  store atomic i64 %3, i64* %floc release, align 8
  ret void
}

@glob32 = global float 0.000000e+00, align 4
@glob64 = global double 0.000000e+00, align 8

; Floating-point add to a global using an immediate.
define void @fadd_32g() {
; X64-LABEL: fadd_32g:
; X64-NOT: lock
; X64:      movss .{{[A-Z0-9_]+}}(%rip), %[[XMM:xmm[0-9]+]]
; X64-NEXT: addss glob32(%rip), %[[XMM]]
; X64-NEXT: movss %[[XMM]], glob32(%rip)
; X32-LABEL: fadd_32g:
; Don't check x86-32 (see comment above).
  %i = load atomic i32, i32* bitcast (float* @glob32 to i32*) monotonic, align 4
  %f = bitcast i32 %i to float
  %add = fadd float %f, 1.000000e+00
  %s = bitcast float %add to i32
  store atomic i32 %s, i32* bitcast (float* @glob32 to i32*) monotonic, align 4
  ret void
}

define void @fadd_64g() {
; X64-LABEL: fadd_64g:
; X64-NOT: lock
; X64:      movsd .{{[A-Z0-9_]+}}(%rip), %[[XMM:xmm[0-9]+]]
; X64-NEXT: addsd glob64(%rip), %[[XMM]]
; X64-NEXT: movsd %[[XMM]], glob64(%rip)
; X32-LABEL: fadd_64g:
; Don't check x86-32 (see comment above).
  %i = load atomic i64, i64* bitcast (double* @glob64 to i64*) monotonic, align 8
  %f = bitcast i64 %i to double
  %add = fadd double %f, 1.000000e+00
  %s = bitcast double %add to i64
  store atomic i64 %s, i64* bitcast (double* @glob64 to i64*) monotonic, align 8
  ret void
}

; Floating-point add to a hard-coded immediate location using an immediate.
define void @fadd_32imm() {
; X64-LABEL: fadd_32imm:
; X64-NOT: lock
; X64:      movl $3735928559, %e[[M:[a-z]+]]
; X64:      movss .{{[A-Z0-9_]+}}(%rip), %[[XMM:xmm[0-9]+]]
; X64-NEXT: addss (%r[[M]]), %[[XMM]]
; X64-NEXT: movss %[[XMM]], (%r[[M]])
; X32-LABEL: fadd_32imm:
; Don't check x86-32 (see comment above).
  %i = load atomic i32, i32* inttoptr (i32 3735928559 to i32*) monotonic, align 4
  %f = bitcast i32 %i to float
  %add = fadd float %f, 1.000000e+00
  %s = bitcast float %add to i32
  store atomic i32 %s, i32* inttoptr (i32 3735928559 to i32*) monotonic, align 4
  ret void
}

define void @fadd_64imm() {
; X64-LABEL: fadd_64imm:
; X64-NOT: lock
; X64:      movl $3735928559, %e[[M:[a-z]+]]
; X64:      movsd .{{[A-Z0-9_]+}}(%rip), %[[XMM:xmm[0-9]+]]
; X64-NEXT: addsd (%r[[M]]), %[[XMM]]
; X64-NEXT: movsd %[[XMM]], (%r[[M]])
; X32-LABEL: fadd_64imm:
; Don't check x86-32 (see comment above).
  %i = load atomic i64, i64* inttoptr (i64 3735928559 to i64*) monotonic, align 8
  %f = bitcast i64 %i to double
  %add = fadd double %f, 1.000000e+00
  %s = bitcast double %add to i64
  store atomic i64 %s, i64* inttoptr (i64 3735928559 to i64*) monotonic, align 8
  ret void
}

; Floating-point add to a stack location.
define void @fadd_32stack() {
; X64-LABEL: fadd_32stack:
; X64-NOT: lock
; X64:      movss .{{[A-Z0-9_]+}}(%rip), %[[XMM:xmm[0-9]+]]
; X64-NEXT: addss [[STACKOFF:-?[0-9]+]](%rsp), %[[XMM]]
; X64-NEXT: movss %[[XMM]], [[STACKOFF]](%rsp)
; X32-LABEL: fadd_32stack:
; Don't check x86-32 (see comment above).
  %ptr = alloca i32, align 4
  %bc3 = bitcast i32* %ptr to float*
  %load = load atomic i32, i32* %ptr acquire, align 4
  %bc0 = bitcast i32 %load to float
  %fadd = fadd float 1.000000e+00, %bc0
  %bc1 = bitcast float %fadd to i32
  store atomic i32 %bc1, i32* %ptr release, align 4
  ret void
}

define void @fadd_64stack() {
; X64-LABEL: fadd_64stack:
; X64-NOT: lock
; X64:      movsd .{{[A-Z0-9_]+}}(%rip), %[[XMM:xmm[0-9]+]]
; X64-NEXT: addsd [[STACKOFF:-?[0-9]+]](%rsp), %[[XMM]]
; X64-NEXT: movsd %[[XMM]], [[STACKOFF]](%rsp)
; X32-LABEL: fadd_64stack:
; Don't check x86-32 (see comment above).
  %ptr = alloca i64, align 8
  %bc3 = bitcast i64* %ptr to double*
  %load = load atomic i64, i64* %ptr acquire, align 8
  %bc0 = bitcast i64 %load to double
  %fadd = fadd double 1.000000e+00, %bc0
  %bc1 = bitcast double %fadd to i64
  store atomic i64 %bc1, i64* %ptr release, align 8
  ret void
}

define void @fadd_array(i64* %arg, double %arg1, i64 %arg2) {
; X64-LABEL: fadd_array:
; X64-NOT: lock
; X64: addsd ([[ADDR:%r..,%r..,8]]), %[[XMM:xmm[0-9]+]]
; X64-NEXT: movsd %[[XMM]], ([[ADDR]])
; X32-LABEL: fadd_array:
; Don't check x86-32 (see comment above).
bb:
  %tmp4 = getelementptr inbounds i64, i64* %arg, i64 %arg2
  %tmp6 = load atomic i64, i64* %tmp4 monotonic, align 8
  %tmp7 = bitcast i64 %tmp6 to double
  %tmp8 = fadd double %tmp7, %arg1
  %tmp9 = bitcast double %tmp8 to i64
  store atomic i64 %tmp9, i64* %tmp4 monotonic, align 8
  ret void
}
