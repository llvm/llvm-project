; RUN: opt < %s -passes=instcombine -S | grep "align 32" | count 2

; It's tempting to have an instcombine in which the src pointer of a
; memcpy is aligned up to the alignment of the destination, however
; there are pitfalls. If the src is an alloca, aligning it beyond what
; the target's stack pointer is aligned at will require dynamic
; stack realignment, which can require functions that don't otherwise
; need a frame pointer to need one.
;
; Abstaining from this transform is not the only way to approach this
; issue. Some late phase could be smart enough to reduce alloca
; alignments when they are greater than they need to be. Or, codegen
; could do dynamic alignment for just the one alloca, and leave the
; main stack pointer at its standard alignment.
;


@dst = global [1024 x i8] zeroinitializer, align 32

define void @foo() nounwind {
entry:
  %src = alloca [1024 x i8], align 64
  call void @llvm.memcpy.p0.p0.i32(ptr align 32 @dst, ptr align 32 %src, i32 1024, i1 false)
  call void @frob(ptr %src) nounwind
  ret void
}

declare void @frob(ptr)

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
