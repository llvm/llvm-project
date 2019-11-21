; RUN: llc < %s -march=sparc -mcpu=v9 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=sparcv9 -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: test_atomic_i8
; CHECK:       ldub [%o0]
; CHECK:       membar
; CHECK:       ldub [%o1]
; CHECK:       membar
; CHECK:       membar
; CHECK:       stb {{.+}}, [%o2]
define i8 @test_atomic_i8(i8* %ptr1, i8* %ptr2, i8* %ptr3) {
entry:
  %0 = load atomic i8, i8* %ptr1 acquire, align 1
  %1 = load atomic i8, i8* %ptr2 acquire, align 1
  %2 = add i8 %0, %1
  store atomic i8 %2, i8* %ptr3 release, align 1
  ret i8 %2
}

; CHECK-LABEL: test_atomic_i16
; CHECK:       lduh [%o0]
; CHECK:       membar
; CHECK:       lduh [%o1]
; CHECK:       membar
; CHECK:       membar
; CHECK:       sth {{.+}}, [%o2]
define i16 @test_atomic_i16(i16* %ptr1, i16* %ptr2, i16* %ptr3) {
entry:
  %0 = load atomic i16, i16* %ptr1 acquire, align 2
  %1 = load atomic i16, i16* %ptr2 acquire, align 2
  %2 = add i16 %0, %1
  store atomic i16 %2, i16* %ptr3 release, align 2
  ret i16 %2
}

; CHECK-LABEL: test_atomic_i32
; CHECK:       ld [%o0]
; CHECK:       membar
; CHECK:       ld [%o1]
; CHECK:       membar
; CHECK:       membar
; CHECK:       st {{.+}}, [%o2]
define i32 @test_atomic_i32(i32* %ptr1, i32* %ptr2, i32* %ptr3) {
entry:
  %0 = load atomic i32, i32* %ptr1 acquire, align 4
  %1 = load atomic i32, i32* %ptr2 acquire, align 4
  %2 = add i32 %0, %1
  store atomic i32 %2, i32* %ptr3 release, align 4
  ret i32 %2
}

;; TODO: the "move %icc" and related instructions are totally
;; redundant here. There's something weird happening in optimization
;; of the success value of cmpxchg.

; CHECK-LABEL: test_cmpxchg_i8
; CHECK:       and %o1, -4, %o2
; CHECK:       mov  3, %o3
; CHECK:       andn %o3, %o1, %o1
; CHECK:       sll %o1, 3, %o1
; CHECK:       mov  255, %o3
; CHECK:       sll %o3, %o1, %o5
; CHECK:       xor %o5, -1, %o3
; CHECK:       mov  123, %o4
; CHECK:       ld [%o2], %g2
; CHECK:       sll %o4, %o1, %o4
; CHECK:       and %o0, 255, %o0
; CHECK:       sll %o0, %o1, %o0
; CHECK:       andn %g2, %o5, %g2
; CHECK:       mov %g0, %o5
; CHECK:      [[LABEL1:\.L.*]]:
; CHECK:       or %g2, %o4, %g3
; CHECK:       or %g2, %o0, %g4
; CHECK:       cas [%o2], %g4, %g3
; CHECK:       cmp %g3, %g4
; CHECK:       mov  %o5, %g4
; CHECK:       move %icc, 1, %g4
; CHECK:       cmp %g4, 0
; CHECK:       bne  [[LABEL2:\.L.*]]
; CHECK:       nop
; CHECK:       and %g3, %o3, %g4
; CHECK:       cmp %g2, %g4
; CHECK:       bne  [[LABEL1]]
; CHECK:       mov  %g4, %g2
; CHECK:      [[LABEL2]]:
; CHECK:       retl
; CHECK:       srl %g3, %o1, %o0
define i8 @test_cmpxchg_i8(i8 %a, i8* %ptr) {
entry:
  %pair = cmpxchg i8* %ptr, i8 %a, i8 123 monotonic monotonic
  %b = extractvalue { i8, i1 } %pair, 0
  ret i8 %b
}

; CHECK-LABEL: test_cmpxchg_i16

; CHECK:       and %o1, -4, %o2
; CHECK:       and %o1, 3, %o1
; CHECK:       xor %o1, 2, %o1
; CHECK:       sll %o1, 3, %o1
; CHECK:       sethi 63, %o3
; CHECK:       or %o3, 1023, %o4
; CHECK:       sll %o4, %o1, %o5
; CHECK:       xor %o5, -1, %o3
; CHECK:       and %o0, %o4, %o4
; CHECK:       ld [%o2], %g2
; CHECK:       mov  123, %o0
; CHECK:       sll %o0, %o1, %o0
; CHECK:       sll %o4, %o1, %o4
; CHECK:       andn %g2, %o5, %g2
; CHECK:       mov %g0, %o5
; CHECK:      [[LABEL1:\.L.*]]:
; CHECK:       or %g2, %o0, %g3
; CHECK:       or %g2, %o4, %g4
; CHECK:       cas [%o2], %g4, %g3
; CHECK:       cmp %g3, %g4
; CHECK:       mov  %o5, %g4
; CHECK:       move %icc, 1, %g4
; CHECK:       cmp %g4, 0
; CHECK:       bne  [[LABEL2:\.L.*]]
; CHECK:       nop
; CHECK:       and %g3, %o3, %g4
; CHECK:       cmp %g2, %g4
; CHECK:       bne  [[LABEL1]]
; CHECK:       mov  %g4, %g2
; CHECK:      [[LABEL2]]:
; CHECK:       retl
; CHECK:       srl %g3, %o1, %o0
define i16 @test_cmpxchg_i16(i16 %a, i16* %ptr) {
entry:
  %pair = cmpxchg i16* %ptr, i16 %a, i16 123 monotonic monotonic
  %b = extractvalue { i16, i1 } %pair, 0
  ret i16 %b
}

; CHECK-LABEL: test_cmpxchg_i32
; CHECK:       mov 123, [[R:%[gilo][0-7]]]
; CHECK:       cas [%o1], %o0, [[R]]

define i32 @test_cmpxchg_i32(i32 %a, i32* %ptr) {
entry:
  %pair = cmpxchg i32* %ptr, i32 %a, i32 123 monotonic monotonic
  %b = extractvalue { i32, i1 } %pair, 0
  ret i32 %b
}

; CHECK-LABEL: test_swap_i8
; CHECK:       mov 42, [[R:%[gilo][0-7]]]
; CHECK:       cas

define i8 @test_swap_i8(i8 %a, i8* %ptr) {
entry:
  %b = atomicrmw xchg i8* %ptr, i8 42 monotonic
  ret i8 %b
}

; CHECK-LABEL: test_swap_i16
; CHECK:       mov 42, [[R:%[gilo][0-7]]]
; CHECK:       cas

define i16 @test_swap_i16(i16 %a, i16* %ptr) {
entry:
  %b = atomicrmw xchg i16* %ptr, i16 42 monotonic
  ret i16 %b
}

; CHECK-LABEL: test_swap_i32
; CHECK:       mov 42, [[R:%[gilo][0-7]]]
; CHECK:       swap [%o1], [[R]]

define i32 @test_swap_i32(i32 %a, i32* %ptr) {
entry:
  %b = atomicrmw xchg i32* %ptr, i32 42 monotonic
  ret i32 %b
}

; CHECK-LABEL: test_load_sub_i8
; CHECK: membar
; CHECK: .L{{.*}}:
; CHECK: sub
; CHECK: cas [{{%[gilo][0-7]}}]
; CHECK: membar
define zeroext i8 @test_load_sub_i8(i8* %p, i8 zeroext %v) {
entry:
  %0 = atomicrmw sub i8* %p, i8 %v seq_cst
  ret i8 %0
}

; CHECK-LABEL: test_load_sub_i16
; CHECK: membar
; CHECK: .L{{.*}}:
; CHECK: sub
; CHECK: cas [{{%[gilo][0-7]}}]
; CHECK: membar
define zeroext i16 @test_load_sub_i16(i16* %p, i16 zeroext %v) {
entry:
  %0 = atomicrmw sub i16* %p, i16 %v seq_cst
  ret i16 %0
}

; CHECK-LABEL: test_load_add_i32
; CHECK: membar
; CHECK: mov %g0
; CHECK: mov [[U:%[gilo][0-7]]], [[V:%[gilo][0-7]]]
; CHECK: add [[U:%[gilo][0-7]]], %o1, [[V2:%[gilo][0-7]]]
; CHECK: cas [%o0], [[V]], [[V2]]
; CHECK: membar
define zeroext i32 @test_load_add_i32(i32* %p, i32 zeroext %v) {
entry:
  %0 = atomicrmw add i32* %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK-LABEL: test_load_xor_32
; CHECK: membar
; CHECK: xor
; CHECK: cas [%o0]
; CHECK: membar
define zeroext i32 @test_load_xor_32(i32* %p, i32 zeroext %v) {
entry:
  %0 = atomicrmw xor i32* %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK-LABEL: test_load_and_32
; CHECK: membar
; CHECK: and
; CHECK-NOT: xor
; CHECK: cas [%o0]
; CHECK: membar
define zeroext i32 @test_load_and_32(i32* %p, i32 zeroext %v) {
entry:
  %0 = atomicrmw and i32* %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK-LABEL: test_load_nand_32
; CHECK: membar
; CHECK: and
; CHECK: xor
; CHECK: cas [%o0]
; CHECK: membar
define zeroext i32 @test_load_nand_32(i32* %p, i32 zeroext %v) {
entry:
  %0 = atomicrmw nand i32* %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK-LABEL: test_load_umin_32
; CHECK: membar
; CHECK: cmp
; CHECK: movleu %icc
; CHECK: cas [%o0]
; CHECK: membar
define zeroext i32 @test_load_umin_32(i32* %p, i32 zeroext %v) {
entry:
  %0 = atomicrmw umin i32* %p, i32 %v seq_cst
  ret i32 %0
}
