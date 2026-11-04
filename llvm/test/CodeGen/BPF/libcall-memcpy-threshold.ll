; RUN: llc < %s -mtriple=bpfel -verify-machineinstrs -bpf-max-stores-per-memfunc=1 | FileCheck %s --check-prefix=LIBCALL

; StoresNumEstimate = alignTo(16, 8) >> 3 = 2, which is greater than the
; getCommonMaxStoresPerMemFunc() value of 1 from -bpf-max-stores-per-memfunc.
; That prevents inline expansion and lets the generic memcpy libcall path fire.
define dso_local void @small_copy(ptr nocapture %dst, ptr nocapture readonly %src) local_unnamed_addr {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 16, i1 false)
  ret void
}

; StoresNumEstimate = alignTo(16, 16) >> 4 = 1, so the threshold still allows
; inline expansion here. The memcpy must still avoid BPF::MEMCPY because BPF
; only supports alignment up to 8 bytes.
define dso_local void @align16_copy(ptr nocapture %dst, ptr nocapture readonly %src) local_unnamed_addr {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 16 %dst, ptr align 16 %src, i64 16, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)

; LIBCALL-LABEL: small_copy:
; LIBCALL:       r3 = 16
; LIBCALL:       call memcpy
; LIBCALL-LABEL: align16_copy:
; LIBCALL:       r3 = 16
; LIBCALL:       call memcpy
