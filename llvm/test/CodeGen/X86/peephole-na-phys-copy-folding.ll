; RUN: llc -verify-machineinstrs -mtriple=i386-linux-gnu %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=x86_64-linux-gnu -mattr=+sahf %s -o - | FileCheck %s

; The peephole optimizer can elide some physical register copies such as
; EFLAGS. Make sure the flags are used directly, instead of needlessly using
; lahf, when possible.

@L = external global i32
@M = external global i8
declare i32 @bar(i64)

; CHECK-LABEL: plus_one
; CHECK-NOT: seto
; CHECK-NOT: lahf
; CHECK-NOT: sahf
; CHECK-NOT: pushf
; CHECK-NOT: popf
; CHECK: incl L
define i1 @plus_one() {
entry:
  %loaded_L = load i32, i32* @L
  %val = add nsw i32 %loaded_L, 1 ; N.B. will emit inc.
  store i32 %val, i32* @L
  %loaded_M = load i8, i8* @M
  %masked = and i8 %loaded_M, 8
  %M_is_true = icmp ne i8 %masked, 0
  %L_is_false = icmp eq i32 %val, 0
  %cond = and i1 %L_is_false, %M_is_true
  br i1 %cond, label %exit2, label %exit

exit:
  ret i1 true

exit2:
  ret i1 false
}

; CHECK-LABEL: plus_forty_two
; CHECK-NOT: seto
; CHECK-NOT: lahf
; CHECK-NOT: sahf
; CHECK-NOT: pushf
; CHECK-NOT: popf
; CHECK: addl $42,
define i1 @plus_forty_two() {
entry:
  %loaded_L = load i32, i32* @L
  %val = add nsw i32 %loaded_L, 42 ; N.B. won't emit inc.
  store i32 %val, i32* @L
  %loaded_M = load i8, i8* @M
  %masked = and i8 %loaded_M, 8
  %M_is_true = icmp ne i8 %masked, 0
  %L_is_false = icmp eq i32 %val, 0
  %cond = and i1 %L_is_false, %M_is_true
  br i1 %cond, label %exit2, label %exit

exit:
  ret i1 true

exit2:
  ret i1 false
}

; CHECK-LABEL: minus_one
; CHECK-NOT: seto
; CHECK-NOT: lahf
; CHECK-NOT: sahf
; CHECK-NOT: pushf
; CHECK-NOT: popf
; CHECK: decl L
define i1 @minus_one() {
entry:
  %loaded_L = load i32, i32* @L
  %val = add nsw i32 %loaded_L, -1 ; N.B. will emit dec.
  store i32 %val, i32* @L
  %loaded_M = load i8, i8* @M
  %masked = and i8 %loaded_M, 8
  %M_is_true = icmp ne i8 %masked, 0
  %L_is_false = icmp eq i32 %val, 0
  %cond = and i1 %L_is_false, %M_is_true
  br i1 %cond, label %exit2, label %exit

exit:
  ret i1 true

exit2:
  ret i1 false
}

; CHECK-LABEL: minus_forty_two
; CHECK-NOT: seto
; CHECK-NOT: lahf
; CHECK-NOT: sahf
; CHECK-NOT: pushf
; CHECK-NOT: popf
; CHECK: addl $-42,
define i1 @minus_forty_two() {
entry:
  %loaded_L = load i32, i32* @L
  %val = add nsw i32 %loaded_L, -42 ; N.B. won't emit dec.
  store i32 %val, i32* @L
  %loaded_M = load i8, i8* @M
  %masked = and i8 %loaded_M, 8
  %M_is_true = icmp ne i8 %masked, 0
  %L_is_false = icmp eq i32 %val, 0
  %cond = and i1 %L_is_false, %M_is_true
  br i1 %cond, label %exit2, label %exit

exit:
  ret i1 true

exit2:
  ret i1 false
}

; CHECK-LABEL: test_intervening_call:
; CHECK:       cmpxchg
; CHECK:       seto %al
; CHECK-NEXT:  lahf
; CHECK:       call{{[lq]}} bar
; CHECK:       addb $127, %al
; CHECK-NEXT:  sahf
define i64 @test_intervening_call(i64* %foo, i64 %bar, i64 %baz) {
  ; cmpxchg sets EFLAGS, call clobbers it, then br uses EFLAGS.
  %cx = cmpxchg i64* %foo, i64 %bar, i64 %baz seq_cst seq_cst
  %v = extractvalue { i64, i1 } %cx, 0
  %p = extractvalue { i64, i1 } %cx, 1
  call i32 @bar(i64 %v)
  br i1 %p, label %t, label %f

t:
  ret i64 42

f:
  ret i64 0
}

; CHECK-LABEL: test_two_live_flags:
; CHECK:       cmpxchg
; CHECK:       seto %al
; CHECK-NEXT:  lahf
; Save result of the first cmpxchg into D.
; CHECK-NEXT:  mov{{[lq]}} %[[AX:[er]ax]], %[[D:[re]d[xi]]]
; CHECK:       cmpxchg
; CHECK-NEXT:  sete %al
; Save result of the second cmpxchg onto the stack.
; CHECK-NEXT:  push{{[lq]}} %[[AX]]
; Restore result of the first cmpxchg from D, put it back in EFLAGS.
; CHECK-NEXT:  mov{{[lq]}} %[[D]], %[[AX]]
; CHECK-NEXT:  addb $127, %al
; CHECK-NEXT:  sahf
; Restore result of the second cmpxchg from the stack.
; CHECK-NEXT:  pop{{[lq]}} %[[AX]]
; Test from EFLAGS restored from first cmpxchg, jump if that fails.
; CHECK-NEXT:  jne
; Fallthrough to test the second cmpxchg's result.
; CHECK:       testb %al, %al
; CHECK-NEXT:  je
define i64 @test_two_live_flags(
       i64* %foo0, i64 %bar0, i64 %baz0,
       i64* %foo1, i64 %bar1, i64 %baz1) {
  %cx0 = cmpxchg i64* %foo0, i64 %bar0, i64 %baz0 seq_cst seq_cst
  %p0 = extractvalue { i64, i1 } %cx0, 1
  %cx1 = cmpxchg i64* %foo1, i64 %bar1, i64 %baz1 seq_cst seq_cst
  %p1 = extractvalue { i64, i1 } %cx1, 1
  %flag = and i1 %p0, %p1
  br i1 %flag, label %t, label %f

t:
  ret i64 42

f:
  ret i64 0
}

; CHECK-LABEL: asm_clobbering_flags:
; CHECK:       test
; CHECK-NEXT:  setg
; CHECK-NEXT:  #APP
; CHECK-NEXT:  bsfl
; CHECK-NEXT:  #NO_APP
; CHECK-NEXT:  movl
; CHECK-NEXT:  ret
define i1 @asm_clobbering_flags(i32* %mem) {
  %val = load i32, i32* %mem, align 4
  %cmp = icmp sgt i32 %val, 0
  %res = tail call i32 asm "bsfl $1,$0", "=r,r,~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %val)
  store i32 %res, i32* %mem, align 4
  ret i1 %cmp
}
