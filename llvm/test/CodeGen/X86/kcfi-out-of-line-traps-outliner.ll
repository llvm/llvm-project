; RUN: llc -mtriple=x86_64-unknown-linux-gnu -verify-machineinstrs -enable-machine-outliner < %s | FileCheck %s --check-prefix=OUTLINED

;; KCFI_CMP and KCFI_TRAP are never outlined by the machine outliner, preserving
;; patchable-function-prefix offsets and distinct trap sites. (Kept separate from
;; kcfi-out-of-line-traps.ll because -enable-machine-outliner hits a
;; pre-existing RetpolineThunkInserter assertion on retpoline calls.)
define void @f1(ptr noundef %x, ptr %p) minsize "patchable-function-prefix"="4" {
; OUTLINED-LABEL: f1:
; OUTLINED:         callq OUTLINED_FUNCTION_0
; OUTLINED-NEXT:    movl $4282621618, %r10d
; OUTLINED-NEXT:    addl -8(%rdi), %r10d
; OUTLINED-NEXT:    jne [[TRAP:.LBB[0-9_]+]]
; OUTLINED:       [[TRAP]]:
; OUTLINED:         ud2
  store volatile i64 1, ptr %p, align 8
  %p1 = getelementptr i64, ptr %p, i64 1
  store volatile i64 2, ptr %p1, align 8
  %p2 = getelementptr i64, ptr %p, i64 2
  store volatile i64 3, ptr %p2, align 8
  %p3 = getelementptr i64, ptr %p, i64 3
  store volatile i64 4, ptr %p3, align 8
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

define void @f2(ptr noundef %x, ptr %p) minsize "patchable-function-prefix"="4" {
; OUTLINED-LABEL: f2:
; OUTLINED:         callq OUTLINED_FUNCTION_0
; OUTLINED-NEXT:    movl $4282621618, %r10d
; OUTLINED-NEXT:    addl -8(%rdi), %r10d
; OUTLINED-NEXT:    jne [[TRAP:.LBB[0-9_]+]]
; OUTLINED:       [[TRAP]]:
; OUTLINED:         ud2
; OUTLINED-LABEL: OUTLINED_FUNCTION_0:
; OUTLINED-NOT:     addl
; OUTLINED-NOT:     ud2
; OUTLINED:         retq
; OUTLINED-NOT:   OUTLINED_FUNCTION_1:
  store volatile i64 1, ptr %p, align 8
  %p1 = getelementptr i64, ptr %p, i64 1
  store volatile i64 2, ptr %p1, align 8
  %p2 = getelementptr i64, ptr %p, i64 2
  store volatile i64 3, ptr %p2, align 8
  %p3 = getelementptr i64, ptr %p, i64 3
  store volatile i64 4, ptr %p3, align 8
  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 8, !"kcfi-out-of-line-traps", i32 1}
