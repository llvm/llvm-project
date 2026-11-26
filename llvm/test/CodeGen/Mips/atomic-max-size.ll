; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips1 < %s | FileCheck %s --check-prefix=MIPS1
; RUN: llc -mtriple=mipsel-linux-gnu -mcpu=mips2 < %s | FileCheck %s --check-prefix=O32
; RUN: llc -mtriple=mips-linux-gnu -mcpu=mips64r2 < %s | FileCheck %s --check-prefix=O32
; RUN: llc -mtriple=mips64-linux-gnuabi64 -mcpu=mips64r2 < %s | FileCheck %s --check-prefix=N64

; MIPS I has no ll and sc, so all atomic operations are libcalls. Like Clang,
; 64-bit atomic operations are only done inline for the N32 and N64 ABIs.

define i32 @rmw32(ptr %p, i32 %v) {
; MIPS1-LABEL: rmw32:
; MIPS1:         jal __atomic_fetch_add_4
; O32-LABEL:   rmw32:
; O32:           ll
; O32:           sc
; N64-LABEL:   rmw32:
; N64:           ll
; N64:           sc
  %r = atomicrmw add ptr %p, i32 %v seq_cst
  ret i32 %r
}

define i32 @cas32(ptr %p, i32 %o, i32 %n) {
; MIPS1-LABEL: cas32:
; MIPS1:         jal __atomic_compare_exchange_4
; O32-LABEL:   cas32:
; O32:           ll
; O32:           sc
  %r = cmpxchg ptr %p, i32 %o, i32 %n seq_cst seq_cst
  %v = extractvalue { i32, i1 } %r, 0
  ret i32 %v
}

define i32 @load32(ptr %p) {
; MIPS1-LABEL: load32:
; MIPS1:         jal __atomic_load_4
; O32-LABEL:   load32:
; O32-NOT:       jal
; O32:           lw
  %r = load atomic i32, ptr %p seq_cst, align 4
  ret i32 %r
}

define i64 @rmw64(ptr %p, i64 %v) {
; MIPS1-LABEL: rmw64:
; MIPS1:         jal __atomic_fetch_add_8
; O32-LABEL:   rmw64:
; O32:           jal __atomic_fetch_add_8
; N64-LABEL:   rmw64:
; N64:           lld
; N64:           scd
  %r = atomicrmw add ptr %p, i64 %v seq_cst
  ret i64 %r
}
