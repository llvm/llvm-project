; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

; The fundamental problem: an add separated from other arithmetic by a sext can't
; be combined with the later instructions. However, if the first add is 'nsw',
; then we can promote the sext ahead of that add to allow optimizations.

define i64 @add_nsw_consts(i32 %i) {
; CHECK-LABEL: add_nsw_consts:
; CHECK:       # BB#0:
; CHECK-NEXT:    movslq %edi, %rax
; CHECK-NEXT:    addq $12, %rax
; CHECK-NEXT:    retq

  %add = add nsw i32 %i, 5
  %ext = sext i32 %add to i64
  %idx = add i64 %ext, 7
  ret i64 %idx
}

; An x86 bonus: If we promote the sext ahead of the 'add nsw',
; we allow LEA formation and eliminate an add instruction.

define i64 @add_nsw_sext_add(i32 %i, i64 %x) {
; CHECK-LABEL: add_nsw_sext_add:
; CHECK:       # BB#0:
; CHECK-NEXT:    movslq %edi, %rax
; CHECK-NEXT:    leaq 5(%rax,%rsi), %rax
; CHECK-NEXT:    retq

  %add = add nsw i32 %i, 5
  %ext = sext i32 %add to i64
  %idx = add i64 %x, %ext
  ret i64 %idx
}

; Throw in a scale (left shift) because an LEA can do that too.
; Use a negative constant (LEA displacement) to verify that's handled correctly.

define i64 @add_nsw_sext_lsh_add(i32 %i, i64 %x) {
; CHECK-LABEL: add_nsw_sext_lsh_add:
; CHECK:       # BB#0:
; CHECK-NEXT:    movslq %edi, %rax
; CHECK-NEXT:    leaq -40(%rsi,%rax,8), %rax
; CHECK-NEXT:    retq

  %add = add nsw i32 %i, -5
  %ext = sext i32 %add to i64
  %shl = shl i64 %ext, 3
  %idx = add i64 %x, %shl
  ret i64 %idx
}

; Don't promote the sext if it has no users. The wider add instruction needs an
; extra byte to encode.

define i64 @add_nsw_sext(i32 %i, i64 %x) {
; CHECK-LABEL: add_nsw_sext:
; CHECK:       # BB#0:
; CHECK-NEXT:    addl $5, %edi
; CHECK-NEXT:    movslq %edi, %rax
; CHECK-NEXT:    retq

  %add = add nsw i32 %i, 5
  %ext = sext i32 %add to i64
  ret i64 %ext
}

; The typical use case: a 64-bit system where an 'int' is used as an index into an array.

define i8* @gep8(i32 %i, i8* %x) {
; CHECK-LABEL: gep8:
; CHECK:       # BB#0:
; CHECK-NEXT:    movslq %edi, %rax
; CHECK-NEXT:    leaq 5(%rax,%rsi), %rax
; CHECK-NEXT:    retq

  %add = add nsw i32 %i, 5
  %ext = sext i32 %add to i64
  %idx = getelementptr i8, i8* %x, i64 %ext
  ret i8* %idx
}

define i16* @gep16(i32 %i, i16* %x) {
; CHECK-LABEL: gep16:
; CHECK:       # BB#0:
; CHECK-NEXT:    movslq %edi, %rax
; CHECK-NEXT:    leaq -10(%rsi,%rax,2), %rax
; CHECK-NEXT:    retq

  %add = add nsw i32 %i, -5
  %ext = sext i32 %add to i64
  %idx = getelementptr i16, i16* %x, i64 %ext
  ret i16* %idx
}

define i32* @gep32(i32 %i, i32* %x) {
; CHECK-LABEL: gep32:
; CHECK:       # BB#0:
; CHECK-NEXT:    movslq %edi, %rax
; CHECK-NEXT:    leaq 20(%rsi,%rax,4), %rax
; CHECK-NEXT:    retq

  %add = add nsw i32 %i, 5
  %ext = sext i32 %add to i64
  %idx = getelementptr i32, i32* %x, i64 %ext
  ret i32* %idx
}

define i64* @gep64(i32 %i, i64* %x) {
; CHECK-LABEL: gep64:
; CHECK:       # BB#0:
; CHECK-NEXT:    movslq %edi, %rax
; CHECK-NEXT:    leaq -40(%rsi,%rax,8), %rax
; CHECK-NEXT:    retq

  %add = add nsw i32 %i, -5
  %ext = sext i32 %add to i64
  %idx = getelementptr i64, i64* %x, i64 %ext
  ret i64* %idx
}

; LEA can't scale by 16, but the adds can still be combined into an LEA.

define i128* @gep128(i32 %i, i128* %x) {
; CHECK-LABEL: gep128:
; CHECK:       # BB#0:
; CHECK-NEXT:    movslq %edi, %rax
; CHECK-NEXT:    shlq $4, %rax
; CHECK-NEXT:    leaq 80(%rax,%rsi), %rax
; CHECK-NEXT:    retq

  %add = add nsw i32 %i, 5
  %ext = sext i32 %add to i64
  %idx = getelementptr i128, i128* %x, i64 %ext
  ret i128* %idx
}

; A bigger win can be achieved when there is more than one use of the
; sign extended value. In this case, we can eliminate sign extension
; instructions plus use more efficient addressing modes for memory ops.

define void @PR20134(i32* %a, i32 %i) {
; CHECK-LABEL: PR20134:
; CHECK:       # BB#0:
; CHECK-NEXT:    movslq %esi, %rax
; CHECK-NEXT:    movl 4(%rdi,%rax,4), %ecx
; CHECK-NEXT:    addl 8(%rdi,%rax,4), %ecx
; CHECK-NEXT:    movl %ecx, (%rdi,%rax,4)
; CHECK-NEXT:    retq

  %add1 = add nsw i32 %i, 1
  %idx1 = sext i32 %add1 to i64
  %gep1 = getelementptr i32, i32* %a, i64 %idx1
  %load1 = load i32, i32* %gep1, align 4

  %add2 = add nsw i32 %i, 2
  %idx2 = sext i32 %add2 to i64
  %gep2 = getelementptr i32, i32* %a, i64 %idx2
  %load2 = load i32, i32* %gep2, align 4

  %add3 = add i32 %load1, %load2
  %idx3 = sext i32 %i to i64
  %gep3 = getelementptr i32, i32* %a, i64 %idx3
  store i32 %add3, i32* %gep3, align 4
  ret void
}

