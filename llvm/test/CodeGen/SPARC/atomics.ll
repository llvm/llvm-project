; RUN: llc < %s -mtriple=sparc -mcpu=v9 -verify-machineinstrs | FileCheck %s --check-prefixes=SPARC
; RUN: llc < %s -mtriple=sparcv9 -verify-machineinstrs | FileCheck %s --check-prefixes=SPARC64

; SPARC-LABEL: test_atomic_i8
; SPARC:       ldub [%o0]
; SPARC:       membar
; SPARC:       ldub [%o1]
; SPARC:       membar
; SPARC:       membar
; SPARC:       stb {{.+}}, [%o2]
; SPARC64-LABEL: test_atomic_i8
; SPARC64:       ldub [%o0]
; SPARC64:       membar
; SPARC64:       ldub [%o1]
; SPARC64:       membar
; SPARC64:       membar
; SPARC64:       stb {{.+}}, [%o2]
define i8 @test_atomic_i8(ptr %ptr1, ptr %ptr2, ptr %ptr3) {
entry:
  %0 = load atomic i8, ptr %ptr1 acquire, align 1
  %1 = load atomic i8, ptr %ptr2 acquire, align 1
  %2 = add i8 %0, %1
  store atomic i8 %2, ptr %ptr3 release, align 1
  ret i8 %2
}

; SPARC-LABEL: test_atomic_i16
; SPARC:       lduh [%o0]
; SPARC:       membar
; SPARC:       lduh [%o1]
; SPARC:       membar
; SPARC:       membar
; SPARC:       sth {{.+}}, [%o2]
; SPARC64-LABEL: test_atomic_i16
; SPARC64:       lduh [%o0]
; SPARC64:       membar
; SPARC64:       lduh [%o1]
; SPARC64:       membar
; SPARC64:       membar
; SPARC64:       sth {{.+}}, [%o2]
define i16 @test_atomic_i16(ptr %ptr1, ptr %ptr2, ptr %ptr3) {
entry:
  %0 = load atomic i16, ptr %ptr1 acquire, align 2
  %1 = load atomic i16, ptr %ptr2 acquire, align 2
  %2 = add i16 %0, %1
  store atomic i16 %2, ptr %ptr3 release, align 2
  ret i16 %2
}

; SPARC-LABEL: test_atomic_i32
; SPARC:       ld [%o0]
; SPARC:       membar
; SPARC:       ld [%o1]
; SPARC:       membar
; SPARC:       membar
; SPARC:       st {{.+}}, [%o2]
; SPARC64-LABEL: test_atomic_i32
; SPARC64:       ld [%o0]
; SPARC64:       membar
; SPARC64:       ld [%o1]
; SPARC64:       membar
; SPARC64:       membar
; SPARC64:       st {{.+}}, [%o2]
define i32 @test_atomic_i32(ptr %ptr1, ptr %ptr2, ptr %ptr3) {
entry:
  %0 = load atomic i32, ptr %ptr1 acquire, align 4
  %1 = load atomic i32, ptr %ptr2 acquire, align 4
  %2 = add i32 %0, %1
  store atomic i32 %2, ptr %ptr3 release, align 4
  ret i32 %2
}

;; TODO: the "move %icc" and related instructions are totally
;; redundant here. There's something weird happening in optimization
;; of the success value of cmpxchg.

; SPARC-LABEL: test_cmpxchg_i8
; SPARC:       and %o1, -4, %o2
; SPARC:       mov  3, %o3
; SPARC:       andn %o3, %o1, %o1
; SPARC:       sll %o1, 3, %o1
; SPARC:       mov  255, %o3
; SPARC:       sll %o3, %o1, %o5
; SPARC:       xor %o5, -1, %o3
; SPARC:       mov  123, %o4
; SPARC:       ld [%o2], %g2
; SPARC:       sll %o4, %o1, %o4
; SPARC:       and %o0, 255, %o0
; SPARC:       sll %o0, %o1, %o0
; SPARC:       andn %g2, %o5, %o5
; SPARC:      [[LABEL1:\.L.*]]:
; SPARC:       or %o5, %o4, %g2
; SPARC:       or %o5, %o0, %g3
; SPARC:       cas [%o2], %g3, %g2
; SPARC:       mov %g0, %g4
; SPARC:       cmp %g2, %g3
; SPARC:       move %icc, 1, %g4
; SPARC:       cmp %g4, 0
; SPARC:       bne %icc, [[LABEL2:\.L.*]]
; SPARC:       nop
; SPARC:       and %g2, %o3, %g3
; SPARC:       cmp %o5, %g3
; SPARC:       bne %icc, [[LABEL1]]
; SPARC:       mov  %g3, %o5
; SPARC:      [[LABEL2]]:
; SPARC:       retl
; SPARC:       srl %g2, %o1, %o0
; SPARC64-LABEL: test_cmpxchg_i8
; SPARC64:       and %o1, -4, %o2
; SPARC64:       mov  3, %o3
; SPARC64:       andn %o3, %o1, %o1
; SPARC64:       sll %o1, 3, %o1
; SPARC64:       mov  255, %o3
; SPARC64:       sll %o3, %o1, %o5
; SPARC64:       xor %o5, -1, %o3
; SPARC64:       mov  123, %o4
; SPARC64:       ld [%o2], %g2
; SPARC64:       sll %o4, %o1, %o4
; SPARC64:       and %o0, 255, %o0
; SPARC64:       sll %o0, %o1, %o0
; SPARC64:       andn %g2, %o5, %o5
; SPARC64:      [[LABEL1:\.L.*]]:
; SPARC64:       or %o5, %o4, %g2
; SPARC64:       or %o5, %o0, %g3
; SPARC64:       cas [%o2], %g3, %g2
; SPARC64:       mov %g0, %g4
; SPARC64:       cmp %g2, %g3
; SPARC64:       move %icc, 1, %g4
; SPARC64:       cmp %g4, 0
; SPARC64:       bne %icc, [[LABEL2:\.L.*]]
; SPARC64:       nop
; SPARC64:       and %g2, %o3, %g3
; SPARC64:       cmp %o5, %g3
; SPARC64:       bne %icc, [[LABEL1]]
; SPARC64:       mov  %g3, %o5
; SPARC64:      [[LABEL2]]:
; SPARC64:       retl
; SPARC64:       srl %g2, %o1, %o0
define i8 @test_cmpxchg_i8(i8 %a, ptr %ptr) {
entry:
  %pair = cmpxchg ptr %ptr, i8 %a, i8 123 monotonic monotonic
  %b = extractvalue { i8, i1 } %pair, 0
  ret i8 %b
}

; SPARC-LABEL: test_cmpxchg_i16
; SPARC:       and %o1, -4, %o2
; SPARC:       and %o1, 3, %o1
; SPARC:       xor %o1, 2, %o1
; SPARC:       sll %o1, 3, %o1
; SPARC:       sethi 63, %o3
; SPARC:       or %o3, 1023, %o4
; SPARC:       sll %o4, %o1, %o5
; SPARC:       xor %o5, -1, %o3
; SPARC:       and %o0, %o4, %o4
; SPARC:       ld [%o2], %g2
; SPARC:       mov  123, %o0
; SPARC:       sll %o0, %o1, %o0
; SPARC:       sll %o4, %o1, %o4
; SPARC:       andn %g2, %o5, %o5
; SPARC:      [[LABEL1:\.L.*]]:
; SPARC:       or %o5, %o0, %g2
; SPARC:       or %o5, %o4, %g3
; SPARC:       cas [%o2], %g3, %g2
; SPARC:       mov %g0, %g4
; SPARC:       cmp %g2, %g3
; SPARC:       move %icc, 1, %g4
; SPARC:       cmp %g4, 0
; SPARC:       bne %icc, [[LABEL2:\.L.*]]
; SPARC:       nop
; SPARC:       and %g2, %o3, %g3
; SPARC:       cmp %o5, %g3
; SPARC:       bne %icc, [[LABEL1]]
; SPARC:       mov  %g3, %o5
; SPARC:      [[LABEL2]]:
; SPARC:       retl
; SPARC:       srl %g2, %o1, %o0
; SPARC64-LABEL: test_cmpxchg_i16
; SPARC64:       and %o1, -4, %o2
; SPARC64:       and %o1, 3, %o1
; SPARC64:       xor %o1, 2, %o1
; SPARC64:       sll %o1, 3, %o1
; SPARC64:       sethi 63, %o3
; SPARC64:       or %o3, 1023, %o4
; SPARC64:       sll %o4, %o1, %o5
; SPARC64:       xor %o5, -1, %o3
; SPARC64:       and %o0, %o4, %o4
; SPARC64:       ld [%o2], %g2
; SPARC64:       mov  123, %o0
; SPARC64:       sll %o0, %o1, %o0
; SPARC64:       sll %o4, %o1, %o4
; SPARC64:       andn %g2, %o5, %o5
; SPARC64:      [[LABEL1:\.L.*]]:
; SPARC64:       or %o5, %o0, %g2
; SPARC64:       or %o5, %o4, %g3
; SPARC64:       cas [%o2], %g3, %g2
; SPARC64:       mov %g0, %g4
; SPARC64:       cmp %g2, %g3
; SPARC64:       move %icc, 1, %g4
; SPARC64:       cmp %g4, 0
; SPARC64:       bne %icc, [[LABEL2:\.L.*]]
; SPARC64:       nop
; SPARC64:       and %g2, %o3, %g3
; SPARC64:       cmp %o5, %g3
; SPARC64:       bne %icc, [[LABEL1]]
; SPARC64:       mov  %g3, %o5
; SPARC64:      [[LABEL2]]:
; SPARC64:       retl
; SPARC64:       srl %g2, %o1, %o0
define i16 @test_cmpxchg_i16(i16 %a, ptr %ptr) {
entry:
  %pair = cmpxchg ptr %ptr, i16 %a, i16 123 monotonic monotonic
  %b = extractvalue { i16, i1 } %pair, 0
  ret i16 %b
}

; SPARC-LABEL: test_cmpxchg_i32
; SPARC:       mov 123, [[R:%[gilo][0-7]]]
; SPARC:       cas [%o1], %o0, [[R]]
; SPARC64-LABEL: test_cmpxchg_i32
; SPARC64:       mov 123, [[R:%[gilo][0-7]]]
; SPARC64:       cas [%o1], %o0, [[R]]
define i32 @test_cmpxchg_i32(i32 %a, ptr %ptr) {
entry:
  %pair = cmpxchg ptr %ptr, i32 %a, i32 123 monotonic monotonic
  %b = extractvalue { i32, i1 } %pair, 0
  ret i32 %b
}

; SPARC-LABEL: test_swap_i8
; SPARC:       mov 42, [[R:%[gilo][0-7]]]
; SPARC:       cas
; SPARC64-LABEL: test_swap_i8
; SPARC64:       mov 42, [[R:%[gilo][0-7]]]
; SPARC64:       cas
define i8 @test_swap_i8(i8 %a, ptr %ptr) {
entry:
  %b = atomicrmw xchg ptr %ptr, i8 42 monotonic
  ret i8 %b
}

; SPARC-LABEL: test_swap_i16
; SPARC:       mov 42, [[R:%[gilo][0-7]]]
; SPARC:       cas
; SPARC64-LABEL: test_swap_i16
; SPARC64:       mov 42, [[R:%[gilo][0-7]]]
; SPARC64:       cas
define i16 @test_swap_i16(i16 %a, ptr %ptr) {
entry:
  %b = atomicrmw xchg ptr %ptr, i16 42 monotonic
  ret i16 %b
}

; SPARC-LABEL: test_swap_i32
; SPARC:       mov 42, [[R:%[gilo][0-7]]]
; SPARC:       swap [%o1], [[R]]
; SPARC64-LABEL: test_swap_i32
; SPARC64:       mov 42, [[R:%[gilo][0-7]]]
; SPARC64:       swap [%o1], [[R]]
define i32 @test_swap_i32(i32 %a, ptr %ptr) {
entry:
  %b = atomicrmw xchg ptr %ptr, i32 42 monotonic
  ret i32 %b
}

; SPARC-LABEL: test_load_sub_i8
; SPARC: membar
; SPARC: .L{{.*}}:
; SPARC: sub
; SPARC: cas [{{%[gilo][0-7]}}]
; SPARC: membar
; SPARC64-LABEL: test_load_sub_i8
; SPARC64: membar
; SPARC64: .L{{.*}}:
; SPARC64: sub
; SPARC64: cas [{{%[gilo][0-7]}}]
; SPARC64: membar
define zeroext i8 @test_load_sub_i8(ptr %p, i8 zeroext %v) {
entry:
  %0 = atomicrmw sub ptr %p, i8 %v seq_cst
  ret i8 %0
}

; SPARC-LABEL: test_load_sub_i16
; SPARC: membar
; SPARC: .L{{.*}}:
; SPARC: sub
; SPARC: cas [{{%[gilo][0-7]}}]
; SPARC: membar
; SPARC64-LABEL: test_load_sub_i16
; SPARC64: membar
; SPARC64: .L{{.*}}:
; SPARC64: sub
; SPARC64: cas [{{%[gilo][0-7]}}]
; SPARC64: membar
define zeroext i16 @test_load_sub_i16(ptr %p, i16 zeroext %v) {
entry:
  %0 = atomicrmw sub ptr %p, i16 %v seq_cst
  ret i16 %0
}

; SPARC-LABEL: test_load_add_i32
; SPARC: membar
; SPARC: mov [[U:%[gilo][0-7]]], [[V:%[gilo][0-7]]]
; SPARC: add [[U:%[gilo][0-7]]], %o1, [[V2:%[gilo][0-7]]]
; SPARC: cas [%o0], [[V]], [[V2]]
; SPARC: membar
; SPARC64-LABEL: test_load_add_i32
; SPARC64: membar
; SPARC64: mov [[U:%[gilo][0-7]]], [[V:%[gilo][0-7]]]
; SPARC64: add [[U:%[gilo][0-7]]], %o1, [[V2:%[gilo][0-7]]]
; SPARC64: cas [%o0], [[V]], [[V2]]
; SPARC64: membar
define zeroext i32 @test_load_add_i32(ptr %p, i32 zeroext %v) {
entry:
  %0 = atomicrmw add ptr %p, i32 %v seq_cst
  ret i32 %0
}

; SPARC-LABEL: test_load_xor_32
; SPARC: membar
; SPARC: xor
; SPARC: cas [%o0]
; SPARC: membar
; SPARC64-LABEL: test_load_xor_32
; SPARC64: membar
; SPARC64: xor
; SPARC64: cas [%o0]
; SPARC64: membar
define zeroext i32 @test_load_xor_32(ptr %p, i32 zeroext %v) {
entry:
  %0 = atomicrmw xor ptr %p, i32 %v seq_cst
  ret i32 %0
}

; SPARC-LABEL: test_load_and_32
; SPARC: membar
; SPARC: and
; SPARC-NOT: xor
; SPARC: cas [%o0]
; SPARC: membar
; SPARC64-LABEL: test_load_and_32
; SPARC64: membar
; SPARC64: and
; SPARC64-NOT: xor
; SPARC64: cas [%o0]
; SPARC64: membar
define zeroext i32 @test_load_and_32(ptr %p, i32 zeroext %v) {
entry:
  %0 = atomicrmw and ptr %p, i32 %v seq_cst
  ret i32 %0
}

; SPARC-LABEL: test_load_nand_32
; SPARC: membar
; SPARC: and
; SPARC: xor
; SPARC: cas [%o0]
; SPARC: membar
; SPARC64-LABEL: test_load_nand_32
; SPARC64: membar
; SPARC64: and
; SPARC64: xor
; SPARC64: cas [%o0]
; SPARC64: membar
define zeroext i32 @test_load_nand_32(ptr %p, i32 zeroext %v) {
entry:
  %0 = atomicrmw nand ptr %p, i32 %v seq_cst
  ret i32 %0
}

; SPARC-LABEL: test_load_umin_32
; SPARC: membar
; SPARC: cmp
; SPARC: movleu %icc
; SPARC: cas [%o0]
; SPARC: membar
; SPARC64-LABEL: test_load_umin_32
; SPARC64: membar
; SPARC64: cmp
; SPARC64: movleu %icc
; SPARC64: cas [%o0]
; SPARC64: membar
define zeroext i32 @test_load_umin_32(ptr %p, i32 zeroext %v) {
entry:
  %0 = atomicrmw umin ptr %p, i32 %v seq_cst
  ret i32 %0
}
