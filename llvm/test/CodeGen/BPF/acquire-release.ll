; RUN: llc < %s -march=bpfel -mcpu=v4 -verify-machineinstrs -show-mc-encoding | FileCheck %s
;
; Source:
;   char load_acquire_i8(char *p) {
;     return __atomic_load_n(p, __ATOMIC_ACQUIRE);
;   }
;   short load_acquire_i16(short *p) {
;     return __atomic_load_n(p, __ATOMIC_ACQUIRE);
;   }
;   int load_acquire_i32(int *p) {
;     return __atomic_load_n(p, __ATOMIC_ACQUIRE);
;   }
;   long load_acquire_i64(long *p) {
;     return __atomic_load_n(p, __ATOMIC_ACQUIRE);
;   }
;   void store_release_i8(char *p, char v) {
;     __atomic_store_n(p, v, __ATOMIC_RELEASE);
;   }
;   void store_release_i16(short *p, short v) {
;     __atomic_store_n(p, v, __ATOMIC_RELEASE);
;   }
;   void store_release_i32(int *p, int v) {
;     __atomic_store_n(p, v, __ATOMIC_RELEASE);
;   }
;   void store_release_i64(long *p, long v) {
;     __atomic_store_n(p, v, __ATOMIC_RELEASE);
;   }

; CHECK-LABEL: load_acquire_i8
; CHECK: w0 = load_acquire((u8 *)(r1 + 0)) # encoding: [0xd3,0x10,0x00,0x00,0x10,0x00,0x00,0x00]
define dso_local i8 @load_acquire_i8(ptr nocapture noundef readonly %p) local_unnamed_addr {
entry:
  %0 = load atomic i8, ptr %p acquire, align 1
  ret i8 %0
}

; CHECK-LABEL: load_acquire_i16
; CHECK: w0 = load_acquire((u16 *)(r1 + 0)) # encoding: [0xcb,0x10,0x00,0x00,0x10,0x00,0x00,0x00]
define dso_local i16 @load_acquire_i16(ptr nocapture noundef readonly %p) local_unnamed_addr {
entry:
  %0 = load atomic i16, ptr %p acquire, align 2
  ret i16 %0
}

; CHECK-LABEL: load_acquire_i32
; CHECK: w0 = load_acquire((u32 *)(r1 + 0)) # encoding: [0xc3,0x10,0x00,0x00,0x10,0x00,0x00,0x00]
define dso_local i32 @load_acquire_i32(ptr nocapture noundef readonly %p) local_unnamed_addr {
entry:
  %0 = load atomic i32, ptr %p acquire, align 4
  ret i32 %0
}

; CHECK-LABEL: load_acquire_i64
; CHECK: r0 = load_acquire((u64 *)(r1 + 0)) # encoding: [0xdb,0x10,0x00,0x00,0x10,0x00,0x00,0x00]
define dso_local i64 @load_acquire_i64(ptr nocapture noundef readonly %p) local_unnamed_addr {
entry:
  %0 = load atomic i64, ptr %p acquire, align 8
  ret i64 %0
}

; CHECK-LABEL: store_release_i8
; CHECK: store_release((u8 *)(r1 + 0), w2) # encoding: [0xd3,0x21,0x00,0x00,0xb0,0x00,0x00,0x00]
define void @store_release_i8(ptr nocapture noundef writeonly %p,
                              i8 noundef signext %v) local_unnamed_addr {
entry:
  store atomic i8 %v, ptr %p release, align 1
  ret void
}

; CHECK-LABEL: store_release_i16
; CHECK: store_release((u16 *)(r1 + 0), w2) # encoding: [0xcb,0x21,0x00,0x00,0xb0,0x00,0x00,0x00]
define void @store_release_i16(ptr nocapture noundef writeonly %p,
                               i16 noundef signext %v) local_unnamed_addr {
entry:
  store atomic i16 %v, ptr %p release, align 2
  ret void
}

; CHECK-LABEL: store_release_i32
; CHECK: store_release((u32 *)(r1 + 0), w2) # encoding: [0xc3,0x21,0x00,0x00,0xb0,0x00,0x00,0x00]
define void @store_release_i32(ptr nocapture noundef writeonly %p,
                               i32 noundef %v) local_unnamed_addr {
entry:
  store atomic i32 %v, ptr %p release, align 4
  ret void
}

; CHECK-LABEL: store_release_i64
; CHECK: store_release((u64 *)(r1 + 0), r2) # encoding: [0xdb,0x21,0x00,0x00,0xb0,0x00,0x00,0x00]
define void @store_release_i64(ptr nocapture noundef writeonly %p,
                               i64 noundef %v) local_unnamed_addr {
entry:
  store atomic i64 %v, ptr %p release, align 8
  ret void
}
