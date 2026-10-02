; RUN: opt -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output %s 2>&1 | FileCheck %s

; BasicAA must not look through llvm.ptrmask when decomposing a GEP into a
; symbolic byte offset, because ptrmask preserves the underlying object but
; can change the byte address. With %base 2-aligned:
;   %p = %base + 1
;   %q = ptrmask(%p, -2) == %base
;   %r = %q + 1          == %p
; so %p and %r alias.

declare ptr @llvm.ptrmask.p0.i64(ptr, i64)

define i8 @ptrmask_gep_may_alias(ptr align 2 %base) {
; CHECK-LABEL: Function: ptrmask_gep_may_alias
; CHECK: MayAlias: i8* %p, i8* %r
entry:
  %p = getelementptr i8, ptr %base, i64 1
  %q = call ptr @llvm.ptrmask.p0.i64(ptr %p, i64 -2)
  %r = getelementptr i8, ptr %q, i64 1

  store i8 7, ptr %p, align 1
  store i8 42, ptr %r, align 1
  %v = load i8, ptr %p, align 1
  ret i8 %v
}
