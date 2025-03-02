; RUN: llc < %s -mtriple=bpfel -mcpu=v3 -verify-machineinstrs -show-mc-encoding | FileCheck %s
;
; Source:
;   int test_load_add_32(int *p, int v) {
;     return __sync_fetch_and_add(p, v);
;   }
;   int test_load_add_64(long *p, long v) {
;     return __sync_fetch_and_add(p, v);
;   }
;   int test_load_sub_32(int *p, int v) {
;     return __sync_fetch_and_sub(p, v);
;   }
;   int test_load_sub_64(long *p, long v) {
;     return __sync_fetch_and_sub(p, v);
;   }
;   // from https://gcc.gnu.org/onlinedocs/gcc-4.1.1/gcc/Atomic-Builtins.html
;   // __sync_lock_test_and_set() actually does atomic xchg and returns
;   // old contents.
;   int test_xchg_32(int *p, int v) {
;     return __sync_lock_test_and_set(p, v);
;   }
;   int test_xchg_64(long *p, long v) {
;     return __sync_lock_test_and_set(p, v);
;   }
;   int test_cas_32(int *p, int old, int new) {
;     return __sync_val_compare_and_swap(p, old, new);
;   }
;   long test_cas_64(long *p, long old, long new) {
;     return __sync_val_compare_and_swap(p, old, new);
;   }
;   int test_load_and_32(int *p, int v) {
;     return __sync_fetch_and_and(p, v);
;   }
;   int test_load_and_64(long *p, long v) {
;     return __sync_fetch_and_and(p, v);
;   }
;   int test_load_or_32(int *p, int v) {
;     return __sync_fetch_and_or(p, v);
;   }
;   int test_load_or_64(long *p, long v) {
;     return __sync_fetch_and_or(p, v);
;   }
;   int test_load_xor_32(int *p, int v) {
;     return __sync_fetch_and_xor(p, v);
;   }
;   int test_load_xor_64(long *p, long v) {
;     return __sync_fetch_and_xor(p, v);
;   }
;   int test_atomic_xor_32(int *p, int v) {
;     __sync_fetch_and_xor(p, v);
;     return 0;
;   }
;   int test_atomic_xor_64(long *p, long v) {
;     __sync_fetch_and_xor(p, v);
;     return 0;
;   }
;   int test_atomic_and_64(long *p, long v) {
;     __sync_fetch_and_and(p, v);
;     return 0;
;   }
;   int test_atomic_or_64(long *p, long v) {
;     __sync_fetch_and_or(p, v);
;     return 0;
;   }

; CHECK-LABEL: test_load_add_32
; CHECK: w0 = w2
; CHECK: w0 = atomic_fetch_add((u32 *)(r1 + 0), w0)
; CHECK: encoding: [0xc3,0x01,0x00,0x00,0x01,0x00,0x00,0x00]
define dso_local i32 @test_load_add_32(ptr nocapture %p, i32 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw add ptr %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK-LABEL: test_load_add_64
; CHECK: r0 = r2
; CHECK: r0 = atomic_fetch_add((u64 *)(r1 + 0), r0)
; CHECK: encoding: [0xdb,0x01,0x00,0x00,0x01,0x00,0x00,0x00]
define dso_local i32 @test_load_add_64(ptr nocapture %p, i64 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw add ptr %p, i64 %v seq_cst
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_load_sub_32
; CHECK: w0 = w2
; CHECK: w0 = -w0
; CHECK: w0 = atomic_fetch_add((u32 *)(r1 + 0), w0)
; CHECK: encoding: [0xc3,0x01,0x00,0x00,0x01,0x00,0x00,0x00]
define dso_local i32 @test_load_sub_32(ptr nocapture %p, i32 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw sub ptr %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK-LABEL: test_load_sub_64
; CHECK: r0 = r2
; CHECK: r0 = -r0
; CHECK: r0 = atomic_fetch_add((u64 *)(r1 + 0), r0)
; CHECK: encoding: [0xdb,0x01,0x00,0x00,0x01,0x00,0x00,0x00]
define dso_local i32 @test_load_sub_64(ptr nocapture %p, i64 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw sub ptr %p, i64 %v seq_cst
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_xchg_32
; CHECK: w0 = w2
; CHECK: w0 = xchg32_32(r1 + 0, w0)
; CHECK: encoding: [0xc3,0x01,0x00,0x00,0xe1,0x00,0x00,0x00]
define dso_local i32 @test_xchg_32(ptr nocapture %p, i32 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw xchg ptr %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK-LABEL: test_xchg_64
; CHECK: r0 = r2
; CHECK: r0 = xchg_64(r1 + 0, r0)
; CHECK: encoding: [0xdb,0x01,0x00,0x00,0xe1,0x00,0x00,0x00]
define dso_local i32 @test_xchg_64(ptr nocapture %p, i64 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw xchg ptr %p, i64 %v seq_cst
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_cas_32
; CHECK: w0 = w2
; CHECK: w0 = cmpxchg32_32(r1 + 0, w0, w3)
; CHECK: encoding: [0xc3,0x31,0x00,0x00,0xf1,0x00,0x00,0x00]
define dso_local i32 @test_cas_32(ptr nocapture %p, i32 %old, i32 %new) local_unnamed_addr {
entry:
  %0 = cmpxchg ptr %p, i32 %old, i32 %new seq_cst seq_cst
  %1 = extractvalue { i32, i1 } %0, 0
  ret i32 %1
}

; CHECK-LABEL: test_cas_64
; CHECK: r0 = r2
; CHECK: r0 = cmpxchg_64(r1 + 0, r0, r3)
; CHECK: encoding: [0xdb,0x31,0x00,0x00,0xf1,0x00,0x00,0x00]
define dso_local i64 @test_cas_64(ptr nocapture %p, i64 %old, i64 %new) local_unnamed_addr {
entry:
  %0 = cmpxchg ptr %p, i64 %old, i64 %new seq_cst seq_cst
  %1 = extractvalue { i64, i1 } %0, 0
  ret i64 %1
}

; CHECK-LABEL: test_load_and_32
; CHECK: w0 = w2
; CHECK: w0 = atomic_fetch_and((u32 *)(r1 + 0), w0)
; CHECK: encoding: [0xc3,0x01,0x00,0x00,0x51,0x00,0x00,0x00]
define dso_local i32 @test_load_and_32(ptr nocapture %p, i32 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw and ptr %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK-LABEL: test_load_and_64
; CHECK: r0 = r2
; CHECK: r0 = atomic_fetch_and((u64 *)(r1 + 0), r0)
; CHECK: encoding: [0xdb,0x01,0x00,0x00,0x51,0x00,0x00,0x00]
define dso_local i32 @test_load_and_64(ptr nocapture %p, i64 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw and ptr %p, i64 %v seq_cst
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_load_or_32
; CHECK: w0 = w2
; CHECK: w0 = atomic_fetch_or((u32 *)(r1 + 0), w0)
; CHECK: encoding: [0xc3,0x01,0x00,0x00,0x41,0x00,0x00,0x00]
define dso_local i32 @test_load_or_32(ptr nocapture %p, i32 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw or ptr %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK-LABEL: test_load_or_64
; CHECK: r0 = r2
; CHECK: r0 = atomic_fetch_or((u64 *)(r1 + 0), r0)
; CHECK: encoding: [0xdb,0x01,0x00,0x00,0x41,0x00,0x00,0x00]
define dso_local i32 @test_load_or_64(ptr nocapture %p, i64 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw or ptr %p, i64 %v seq_cst
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_load_xor_32
; CHECK: w0 = w2
; CHECK: w0 = atomic_fetch_xor((u32 *)(r1 + 0), w0)
; CHECK: encoding: [0xc3,0x01,0x00,0x00,0xa1,0x00,0x00,0x00]
define dso_local i32 @test_load_xor_32(ptr nocapture %p, i32 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw xor ptr %p, i32 %v seq_cst
  ret i32 %0
}

; CHECK-LABEL: test_load_xor_64
; CHECK: r0 = r2
; CHECK: r0 = atomic_fetch_xor((u64 *)(r1 + 0), r0)
; CHECK: encoding: [0xdb,0x01,0x00,0x00,0xa1,0x00,0x00,0x00]
define dso_local i32 @test_load_xor_64(ptr nocapture %p, i64 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw xor ptr %p, i64 %v seq_cst
  %conv = trunc i64 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: test_atomic_xor_32
; CHECK: w2 = atomic_fetch_xor((u32 *)(r1 + 0), w2)
; CHECK: encoding: [0xc3,0x21,0x00,0x00,0xa1,0x00,0x00,0x00]
; CHECK: w0 = 0
define dso_local i32 @test_atomic_xor_32(ptr nocapture %p, i32 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw xor ptr %p, i32 %v seq_cst
  ret i32 0
}

; CHECK-LABEL: test_atomic_xor_64
; CHECK: atomic_fetch_xor((u64 *)(r1 + 0), r2)
; CHECK: encoding: [0xdb,0x21,0x00,0x00,0xa1,0x00,0x00,0x00]
; CHECK: w0 = 0
define dso_local i32 @test_atomic_xor_64(ptr nocapture %p, i64 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw xor ptr %p, i64 %v seq_cst
  ret i32 0
}

; CHECK-LABEL: test_atomic_and_64
; CHECK: r2 = atomic_fetch_and((u64 *)(r1 + 0), r2)
; CHECK: encoding: [0xdb,0x21,0x00,0x00,0x51,0x00,0x00,0x00]
; CHECK: w0 = 0
define dso_local i32 @test_atomic_and_64(ptr nocapture %p, i64 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw and ptr %p, i64 %v seq_cst
  ret i32 0
}

; CHECK-LABEL: test_atomic_or_64
; CHECK: r2 = atomic_fetch_or((u64 *)(r1 + 0), r2)
; CHECK: encoding: [0xdb,0x21,0x00,0x00,0x41,0x00,0x00,0x00]
; CHECK: w0 = 0
define dso_local i32 @test_atomic_or_64(ptr nocapture %p, i64 %v) local_unnamed_addr {
entry:
  %0 = atomicrmw or ptr %p, i64 %v seq_cst
  ret i32 0
}
