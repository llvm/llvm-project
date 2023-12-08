; RUN: llc < %s -march=bpfel -verify-machineinstrs -bpf-expand-memcpy-in-order | FileCheck %s
; RUN: llc < %s -march=bpfeb -verify-machineinstrs -bpf-expand-memcpy-in-order | FileCheck %s
;
; #define COPY_LEN	9
;
; void cal_align1(ptr a, ptr b)
; {
;   __builtin_memcpy(a, b, COPY_LEN);
; }
;
; void cal_align2(short *a, short *b)
; {
;   __builtin_memcpy(a, b, COPY_LEN);
; }
;
; #undef COPY_LEN
; #define COPY_LEN	19
; void cal_align4(int *a, int *b)
; {
;   __builtin_memcpy(a, b, COPY_LEN);
; }
;
; #undef COPY_LEN
; #define COPY_LEN	27
; void cal_align8(long long *a, long long *b)
; {
;   __builtin_memcpy(a, b, COPY_LEN);
; }

; Function Attrs: nounwind
define dso_local void @cal_align1(ptr nocapture %a, ptr nocapture readonly %b) local_unnamed_addr #0 {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 %a, ptr align 1 %b, i64 9, i1 false)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #1

; CHECK: [[SCRATCH_REG:r[0-9]]] = *(u8 *)([[SRC_REG:r[0-9]]] + 0)
; CHECK: *(u8 *)([[DST_REG:r[0-9]]] + 0) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 1)
; CHECK: *(u8 *)([[DST_REG]] + 1) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 2)
; CHECK: *(u8 *)([[DST_REG]] + 2) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 3)
; CHECK: *(u8 *)([[DST_REG]] + 3) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 4)
; CHECK: *(u8 *)([[DST_REG]] + 4) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 5)
; CHECK: *(u8 *)([[DST_REG]] + 5) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 6)
; CHECK: *(u8 *)([[DST_REG]] + 6) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 7)
; CHECK: *(u8 *)([[DST_REG]] + 7) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 8)
; CHECK: *(u8 *)([[DST_REG]] + 8) = [[SCRATCH_REG]]

; Function Attrs: nounwind
define dso_local void @cal_align2(ptr nocapture %a, ptr nocapture readonly %b) local_unnamed_addr #0 {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 2 %a, ptr align 2 %b, i64 9, i1 false)
  ret void
}
; CHECK: [[SCRATCH_REG:r[0-9]]] = *(u16 *)([[SRC_REG:r[0-9]]] + 0)
; CHECK: *(u16 *)([[DST_REG:r[0-9]]] + 0) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u16 *)([[SRC_REG]] + 2)
; CHECK: *(u16 *)([[DST_REG]] + 2) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u16 *)([[SRC_REG]] + 4)
; CHECK: *(u16 *)([[DST_REG]] + 4) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u16 *)([[SRC_REG]] + 6)
; CHECK: *(u16 *)([[DST_REG]] + 6) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 8)
; CHECK: *(u8 *)([[DST_REG]] + 8) = [[SCRATCH_REG]]

; Function Attrs: nounwind
define dso_local void @cal_align4(ptr nocapture %a, ptr nocapture readonly %b) local_unnamed_addr #0 {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a, ptr align 4 %b, i64 19, i1 false)
  ret void
}
; CHECK: [[SCRATCH_REG:r[0-9]]] = *(u32 *)([[SRC_REG:r[0-9]]] + 0)
; CHECK: *(u32 *)([[DST_REG:r[0-9]]] + 0) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u32 *)([[SRC_REG]] + 4)
; CHECK: *(u32 *)([[DST_REG]] + 4) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u32 *)([[SRC_REG]] + 8)
; CHECK: *(u32 *)([[DST_REG]] + 8) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u32 *)([[SRC_REG]] + 12)
; CHECK: *(u32 *)([[DST_REG]] + 12) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u16 *)([[SRC_REG]] + 16)
; CHECK: *(u16 *)([[DST_REG]] + 16) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 18)
; CHECK: *(u8 *)([[DST_REG]] + 18) = [[SCRATCH_REG]]

; Function Attrs: nounwind
define dso_local void @cal_align8(ptr nocapture %a, ptr nocapture readonly %b) local_unnamed_addr #0 {
entry:
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a, ptr align 8 %b, i64 27, i1 false)
  ret void
}
; CHECK: [[SCRATCH_REG:r[0-9]]] = *(u64 *)([[SRC_REG:r[0-9]]] + 0)
; CHECK: *(u64 *)([[DST_REG:r[0-9]]] + 0) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u64 *)([[SRC_REG]] + 8)
; CHECK: *(u64 *)([[DST_REG]] + 8) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u64 *)([[SRC_REG]] + 16)
; CHECK: *(u64 *)([[DST_REG]] + 16) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u16 *)([[SRC_REG]] + 24)
; CHECK: *(u16 *)([[DST_REG]] + 24) = [[SCRATCH_REG]]
; CHECK: [[SCRATCH_REG]] = *(u8 *)([[SRC_REG]] + 26)
; CHECK: *(u8 *)([[DST_REG]] + 26) = [[SCRATCH_REG]]
