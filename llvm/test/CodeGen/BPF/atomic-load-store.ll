; RUN: llc < %s -march=bpfel -mcpu=v4 -verify-machineinstrs -show-mc-encoding \
; RUN:   | FileCheck -check-prefixes=CHECK-LE %s
; RUN: llc < %s -march=bpfeb -mcpu=v4 -verify-machineinstrs -show-mc-encoding \
; RUN:   | FileCheck -check-prefixes=CHECK-BE %s

; Source:
;   void atomic_load_i8(char *p) {
;     (void)__atomic_load_n(p, __ATOMIC_RELAXED);
;     (void)__atomic_load_n(p, __ATOMIC_ACQUIRE);
;   }
;   void atomic_load_i16(short *p) {
;     (void)__atomic_load_n(p, __ATOMIC_RELAXED);
;     (void)__atomic_load_n(p, __ATOMIC_ACQUIRE);
;   }
;   void atomic_load_i32(int *p) {
;     (void)__atomic_load_n(p, __ATOMIC_RELAXED);
;     (void)__atomic_load_n(p, __ATOMIC_ACQUIRE);
;   }
;   void atomic_load_i64(long *p) {
;     (void)__atomic_load_n(p, __ATOMIC_RELAXED);
;     (void)__atomic_load_n(p, __ATOMIC_ACQUIRE);
;   }
;   void atomic_store_i8(char *p, char v) {
;     __atomic_store_n(p, v, __ATOMIC_RELAXED);
;     __atomic_store_n(p, v, __ATOMIC_RELEASE);
;   }
;   void atomic_store_i16(short *p, short v) {
;     __atomic_store_n(p, v, __ATOMIC_RELAXED);
;     __atomic_store_n(p, v, __ATOMIC_RELEASE);
;   }
;   void atomic_store_i32(int *p, int v) {
;     __atomic_store_n(p, v, __ATOMIC_RELAXED);
;     __atomic_store_n(p, v, __ATOMIC_RELEASE);
;   }
;   void atomic_store_i64(long *p, long v) {
;     __atomic_store_n(p, v, __ATOMIC_RELAXED);
;     __atomic_store_n(p, v, __ATOMIC_RELEASE);
;   }

define dso_local void @atomic_load_i8(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
; CHECK-LABEL: atomic_load_i8
; CHECK-LE:      w2 = *(u8 *)(r1 + 0) # encoding: [0x71,0x12,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-LE-NEXT: w1 = load_acquire((u8 *)(r1 + 0)) # encoding: [0xd3,0x11,0x00,0x00,0x00,0x01,0x00,0x00]
;
; CHECK-BE:      w2 = *(u8 *)(r1 + 0) # encoding: [0x71,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-BE-NEXT: w1 = load_acquire((u8 *)(r1 + 0)) # encoding: [0xd3,0x11,0x00,0x00,0x00,0x00,0x01,0x00]
entry:
  %0 = load atomic i8, ptr %p monotonic, align 1
  %1 = load atomic i8, ptr %p acquire, align 1
  ret void
}

define dso_local void @atomic_load_i16(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
; CHECK-LABEL: atomic_load_i16
; CHECK-LE:      w2 = *(u16 *)(r1 + 0) # encoding: [0x69,0x12,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-LE-NEXT: w1 = load_acquire((u16 *)(r1 + 0)) # encoding: [0xcb,0x11,0x00,0x00,0x00,0x01,0x00,0x00]
;
; CHECK-BE:      w2 = *(u16 *)(r1 + 0) # encoding: [0x69,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-BE-NEXT: w1 = load_acquire((u16 *)(r1 + 0)) # encoding: [0xcb,0x11,0x00,0x00,0x00,0x00,0x01,0x00]
entry:
  %0 = load atomic i16, ptr %p monotonic, align 2
  %1 = load atomic i16, ptr %p acquire, align 2
  ret void
}

define dso_local void @atomic_load_i32(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
; CHECK-LABEL: atomic_load_i32
; CHECK-LE:      w2 = *(u32 *)(r1 + 0) # encoding: [0x61,0x12,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-LE-NEXT: w1 = load_acquire((u32 *)(r1 + 0)) # encoding: [0xc3,0x11,0x00,0x00,0x00,0x01,0x00,0x00]
;
; CHECK-BE:      w2 = *(u32 *)(r1 + 0) # encoding: [0x61,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-BE-NEXT: w1 = load_acquire((u32 *)(r1 + 0)) # encoding: [0xc3,0x11,0x00,0x00,0x00,0x00,0x01,0x00]
entry:
  %0 = load atomic i32, ptr %p monotonic, align 4
  %1 = load atomic i32, ptr %p acquire, align 4
  ret void
}

define dso_local void @atomic_load_i64(ptr nocapture noundef readonly %p) local_unnamed_addr #0 {
; CHECK-LABEL: atomic_load_i64
; CHECK-LE:      r2 = *(u64 *)(r1 + 0) # encoding: [0x79,0x12,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-LE-NEXT: r1 = load_acquire((u64 *)(r1 + 0)) # encoding: [0xdb,0x11,0x00,0x00,0x00,0x01,0x00,0x00]
;
; CHECK-BE:      r2 = *(u64 *)(r1 + 0) # encoding: [0x79,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-BE-NEXT: r1 = load_acquire((u64 *)(r1 + 0)) # encoding: [0xdb,0x11,0x00,0x00,0x00,0x00,0x01,0x00]
entry:
  %0 = load atomic i64, ptr %p monotonic, align 8
  %1 = load atomic i64, ptr %p acquire, align 8
  ret void
}

define dso_local void @atomic_store_i8(ptr nocapture noundef writeonly %p, i8 noundef signext %v) local_unnamed_addr #0 {
; CHECK-LABEL: atomic_store_i8
; CHECK-LE:      *(u8 *)(r1 + 0) = w2 # encoding: [0x73,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-LE-NEXT: store_release((u8 *)(r1 + 0), w2) # encoding: [0xd3,0x21,0x00,0x00,0x10,0x01,0x00,0x00]
;
; CHECK-BE:      *(u8 *)(r1 + 0) = w2 # encoding: [0x73,0x12,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-BE-NEXT: store_release((u8 *)(r1 + 0), w2) # encoding: [0xd3,0x12,0x00,0x00,0x00,0x00,0x01,0x10]
entry:
  store atomic i8 %v, ptr %p monotonic, align 1
  store atomic i8 %v, ptr %p release, align 1
  ret void
}

define dso_local void @atomic_store_i16(ptr nocapture noundef writeonly %p, i16 noundef signext %v) local_unnamed_addr #0 {
; CHECK-LABEL: atomic_store_i16
; CHECK-LE:      *(u16 *)(r1 + 0) = w2 # encoding: [0x6b,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-LE-NEXT: store_release((u16 *)(r1 + 0), w2) # encoding: [0xcb,0x21,0x00,0x00,0x10,0x01,0x00,0x00]
;
; CHECK-BE:      *(u16 *)(r1 + 0) = w2 # encoding: [0x6b,0x12,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-BE-NEXT: store_release((u16 *)(r1 + 0), w2) # encoding: [0xcb,0x12,0x00,0x00,0x00,0x00,0x01,0x10]
entry:
  store atomic i16 %v, ptr %p monotonic, align 2
  store atomic i16 %v, ptr %p release, align 2
  ret void
}

define dso_local void @atomic_store_i32(ptr nocapture noundef writeonly %p, i32 noundef %v) local_unnamed_addr #0 {
; CHECK-LABEL: atomic_store_i32
; CHECK-LE:      *(u32 *)(r1 + 0) = w2 # encoding: [0x63,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-LE-NEXT: store_release((u32 *)(r1 + 0), w2) # encoding: [0xc3,0x21,0x00,0x00,0x10,0x01,0x00,0x00]
;
; CHECK-BE:      *(u32 *)(r1 + 0) = w2 # encoding: [0x63,0x12,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-BE-NEXT: store_release((u32 *)(r1 + 0), w2) # encoding: [0xc3,0x12,0x00,0x00,0x00,0x00,0x01,0x10]
entry:
  store atomic i32 %v, ptr %p monotonic, align 4
  store atomic i32 %v, ptr %p release, align 4
  ret void
}

define dso_local void @atomic_store_i64(ptr nocapture noundef writeonly %p, i64 noundef %v) local_unnamed_addr #0 {
; CHECK-LABEL: atomic_store_i64
; CHECK-LE:      *(u64 *)(r1 + 0) = r2 # encoding: [0x7b,0x21,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-LE-NEXT: store_release((u64 *)(r1 + 0), r2) # encoding: [0xdb,0x21,0x00,0x00,0x10,0x01,0x00,0x00]
;
; CHECK-BE:      *(u64 *)(r1 + 0) = r2 # encoding: [0x7b,0x12,0x00,0x00,0x00,0x00,0x00,0x00]
; CHECK-BE-NEXT: store_release((u64 *)(r1 + 0), r2) # encoding: [0xdb,0x12,0x00,0x00,0x00,0x00,0x01,0x10]
entry:
  store atomic i64 %v, ptr %p monotonic, align 8
  store atomic i64 %v, ptr %p release, align 8
  ret void
}
