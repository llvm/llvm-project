; Check that -mips-compact-branches={never,optimal,always} is honoured for
; direct calls and tail calls on MIPS R6.
; RUN: llc -mtriple=mips -mcpu=mips32r6 -relocation-model=static \
; RUN:   -mips-tail-calls=1 -mips-compact-branches=never -verify-machineinstrs \
; RUN:   < %s | FileCheck %s --check-prefixes=NEVER
; RUN: llc -mtriple=mips -mcpu=mips32r6 -relocation-model=static \
; RUN:   -mips-tail-calls=1 -mips-compact-branches=optimal -verify-machineinstrs \
; RUN:   < %s | FileCheck %s --check-prefixes=COMPACT
; RUN: llc -mtriple=mips -mcpu=mips32r6 -relocation-model=static \
; RUN:   -mips-tail-calls=1 -mips-compact-branches=always -verify-machineinstrs \
; RUN:   < %s | FileCheck %s --check-prefixes=COMPACT
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -relocation-model=static \
; RUN:   -mips-tail-calls=1 -mips-compact-branches=never -verify-machineinstrs \
; RUN:   < %s | FileCheck %s --check-prefixes=NEVER
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -relocation-model=static \
; RUN:   -mips-tail-calls=1 -mips-compact-branches=optimal -verify-machineinstrs \
; RUN:   < %s | FileCheck %s --check-prefixes=COMPACT
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -relocation-model=static \
; RUN:   -mips-tail-calls=1 -mips-compact-branches=always -verify-machineinstrs \
; RUN:   < %s | FileCheck %s --check-prefixes=COMPACT

declare void @f()

define void @call() {
; NEVER-LABEL: call:
; NEVER:         jal f
; NEVER-NEXT:    nop
; NEVER-NOT:     balc
;
; COMPACT-LABEL: call:
; COMPACT:       balc f
; COMPACT-NOT:   jal
  call void @f()
  ret void
}

define void @call_libcall(ptr %p, ptr %q) {
; NEVER-LABEL: call_libcall:
; NEVER:         jal memcpy
; NEVER-NOT:     balc
;
; COMPACT-LABEL: call_libcall:
; COMPACT:       balc memcpy
; COMPACT-NOT:   jal
  call void @llvm.memcpy.p0.p0.i32(ptr %p, ptr %q, i32 1024, i1 false)
  ret void
}

define internal void @local() noinline {
  call void asm sideeffect "", ""()
  ret void
}

define void @tailcall() {
; NEVER-LABEL: tailcall:
; NEVER:         j local
; NEVER-NEXT:    nop
;
; COMPACT-LABEL: tailcall:
; COMPACT:       bc local
  tail call void @local()
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1)
