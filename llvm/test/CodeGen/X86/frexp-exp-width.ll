; RUN: llc -O2 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck --check-prefix=X86 %s

; C frexp/frexpf write the exponent through an int * parameter (32 bits).
; On x86-64 i8/i16 are legal SD types, so a declared narrow exponent
; previously got a 1/2-byte temporary that the 4-byte libcall write
; overflowed, and a narrow result store was folded into the libcall
; destination. The temporary must match the libcall's parameter width and
; the declared result must be materialized from it before being stored.
;
; Offset discriminator: with this frame shape (single pushq prologue) the
; unpatched backend packs the 1-byte temporary for i8 at 7(%rsp) and the
; 2-byte temporary for i16 at 6(%rsp) (1/2-byte alignment); the fixed
; backend allocates a C-int-wide object at the 4-byte aligned offset
; 4(%rsp). The exp8/exp16 checks below pin the aligned offset 4, which is
; what separates the two paths.

declare { float, i64 } @llvm.frexp.f32.i64(float)
declare { float, i8 }  @llvm.frexp.f32.i8(float)
declare { float, i16 } @llvm.frexp.f32.i16(float)
declare { float, float } @llvm.modf.f32(float)

; The 32-bit libcall output must be sign-extended when materializing i64
; (previously the full 8-byte slot was read back).
define i64 @exp64(float %x) {
; X86-LABEL: exp64:
; X86: callq frexpf@PLT
; X86: movslq {{[0-9]+}}(%rsp), %rax
  %r = call { float, i64 } @llvm.frexp.f32.i64(float %x)
  %e = extractvalue { float, i64 } %r, 1
  ret i64 %e
}

; Narrow declared exponent (i8): the temporary is C-int wide (4-byte
; aligned slot at 4(%rsp) for this frame shape; the unpatched backend used
; a 1-byte object at 7(%rsp)), and the declared i8 result is truncated
; from the 32-bit read.
define i8 @exp8(float %x) {
; X86-LABEL: exp8:
; X86: leaq 4(%rsp)
; X86: movzbl 4(%rsp)
  %r = call { float, i8 } @llvm.frexp.f32.i8(float %x)
  %e = extractvalue { float, i8 } %r, 1
  ret i8 %e
}

; Same width argument for i16: C-int-wide temporary (4(%rsp); unpatched:
; 2-byte object at 6(%rsp)), result truncated from the 32-bit read (read back with a plain 32-bit
; load; the unpatched backend used a 16-bit load at 6(%rsp)).
define i16 @exp16(float %x) {
; X86-LABEL: exp16:
; X86: leaq 4(%rsp)
; X86: movl 4(%rsp)
  %r = call { float, i16 } @llvm.frexp.f32.i16(float %x)
  %e = extractvalue { float, i16 } %r, 1
  ret i16 %e
}

@gmant = global float 0.000000e+00
@gexp8 = global i8 0

; The i8 destination does not match the libcall's write width: the store
; must NOT be folded into the libcall output pointer; the full-width
; (byte) store to gexp8 must remain, fed from the truncated 32-bit value.
define void @both8(float %x) {
; X86-LABEL: both8:
; X86: callq frexpf@PLT
; X86: movzbl {{[0-9]+}}(%rsp)
; X86: movq gexp8@GOTPCREL(%rip), [[ADDR:%[a-z0-9]+]]
; X86: movb %{{[a-z0-9]+}}, ([[ADDR]])
  %r = call { float, i8 } @llvm.frexp.f32.i8(float %x)
  %m = extractvalue { float, i8 } %r, 0
  %e = extractvalue { float, i8 } %r, 1
  store float %m, ptr @gmant
  store i8 %e, ptr @gexp8
  ret void
}

@gmodfi = global float 0.000000e+00
@gmodff = global float 0.000000e+00

; MODF (C modf) writes the FP type itself through its output pointer; its
; codegen is unaffected by this change.
define void @mf(float %x) {
; X86-LABEL: mf:
; X86: callq modff@PLT
  %r = call { float, float } @llvm.modf.f32(float %x)
  %i = extractvalue { float, float } %r, 0
  %f = extractvalue { float, float } %r, 1
  store float %i, ptr @gmodfi
  store float %f, ptr @gmodff
  ret void
}
