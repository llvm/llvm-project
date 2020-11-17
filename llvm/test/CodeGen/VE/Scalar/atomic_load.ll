; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test atomic load for all types and all memory order
;;;
;;; Note:
;;;   We test i1/i8/i16/i32/i64/i128/u8/u16/u32/u64/u128.
;;;   We test relaxed, acquire, and seq_cst.

%"struct.std::__1::atomic" = type { %"struct.std::__1::__atomic_base" }
%"struct.std::__1::__atomic_base" = type { %"struct.std::__1::__cxx_atomic_impl" }
%"struct.std::__1::__cxx_atomic_impl" = type { %"struct.std::__1::__cxx_atomic_base_impl" }
%"struct.std::__1::__cxx_atomic_base_impl" = type { i8 }
%"struct.std::__1::atomic.0" = type { %"struct.std::__1::__atomic_base.1" }
%"struct.std::__1::__atomic_base.1" = type { %"struct.std::__1::__atomic_base.2" }
%"struct.std::__1::__atomic_base.2" = type { %"struct.std::__1::__cxx_atomic_impl.3" }
%"struct.std::__1::__cxx_atomic_impl.3" = type { %"struct.std::__1::__cxx_atomic_base_impl.4" }
%"struct.std::__1::__cxx_atomic_base_impl.4" = type { i8 }
%"struct.std::__1::atomic.5" = type { %"struct.std::__1::__atomic_base.6" }
%"struct.std::__1::__atomic_base.6" = type { %"struct.std::__1::__atomic_base.7" }
%"struct.std::__1::__atomic_base.7" = type { %"struct.std::__1::__cxx_atomic_impl.8" }
%"struct.std::__1::__cxx_atomic_impl.8" = type { %"struct.std::__1::__cxx_atomic_base_impl.9" }
%"struct.std::__1::__cxx_atomic_base_impl.9" = type { i8 }
%"struct.std::__1::atomic.10" = type { %"struct.std::__1::__atomic_base.11" }
%"struct.std::__1::__atomic_base.11" = type { %"struct.std::__1::__atomic_base.12" }
%"struct.std::__1::__atomic_base.12" = type { %"struct.std::__1::__cxx_atomic_impl.13" }
%"struct.std::__1::__cxx_atomic_impl.13" = type { %"struct.std::__1::__cxx_atomic_base_impl.14" }
%"struct.std::__1::__cxx_atomic_base_impl.14" = type { i16 }
%"struct.std::__1::atomic.15" = type { %"struct.std::__1::__atomic_base.16" }
%"struct.std::__1::__atomic_base.16" = type { %"struct.std::__1::__atomic_base.17" }
%"struct.std::__1::__atomic_base.17" = type { %"struct.std::__1::__cxx_atomic_impl.18" }
%"struct.std::__1::__cxx_atomic_impl.18" = type { %"struct.std::__1::__cxx_atomic_base_impl.19" }
%"struct.std::__1::__cxx_atomic_base_impl.19" = type { i16 }
%"struct.std::__1::atomic.20" = type { %"struct.std::__1::__atomic_base.21" }
%"struct.std::__1::__atomic_base.21" = type { %"struct.std::__1::__atomic_base.22" }
%"struct.std::__1::__atomic_base.22" = type { %"struct.std::__1::__cxx_atomic_impl.23" }
%"struct.std::__1::__cxx_atomic_impl.23" = type { %"struct.std::__1::__cxx_atomic_base_impl.24" }
%"struct.std::__1::__cxx_atomic_base_impl.24" = type { i32 }
%"struct.std::__1::atomic.25" = type { %"struct.std::__1::__atomic_base.26" }
%"struct.std::__1::__atomic_base.26" = type { %"struct.std::__1::__atomic_base.27" }
%"struct.std::__1::__atomic_base.27" = type { %"struct.std::__1::__cxx_atomic_impl.28" }
%"struct.std::__1::__cxx_atomic_impl.28" = type { %"struct.std::__1::__cxx_atomic_base_impl.29" }
%"struct.std::__1::__cxx_atomic_base_impl.29" = type { i32 }
%"struct.std::__1::atomic.30" = type { %"struct.std::__1::__atomic_base.31" }
%"struct.std::__1::__atomic_base.31" = type { %"struct.std::__1::__atomic_base.32" }
%"struct.std::__1::__atomic_base.32" = type { %"struct.std::__1::__cxx_atomic_impl.33" }
%"struct.std::__1::__cxx_atomic_impl.33" = type { %"struct.std::__1::__cxx_atomic_base_impl.34" }
%"struct.std::__1::__cxx_atomic_base_impl.34" = type { i64 }
%"struct.std::__1::atomic.35" = type { %"struct.std::__1::__atomic_base.36" }
%"struct.std::__1::__atomic_base.36" = type { %"struct.std::__1::__atomic_base.37" }
%"struct.std::__1::__atomic_base.37" = type { %"struct.std::__1::__cxx_atomic_impl.38" }
%"struct.std::__1::__cxx_atomic_impl.38" = type { %"struct.std::__1::__cxx_atomic_base_impl.39" }
%"struct.std::__1::__cxx_atomic_base_impl.39" = type { i64 }
%"struct.std::__1::atomic.40" = type { %"struct.std::__1::__atomic_base.41" }
%"struct.std::__1::__atomic_base.41" = type { %"struct.std::__1::__atomic_base.42" }
%"struct.std::__1::__atomic_base.42" = type { %"struct.std::__1::__cxx_atomic_impl.43" }
%"struct.std::__1::__cxx_atomic_impl.43" = type { %"struct.std::__1::__cxx_atomic_base_impl.44" }
%"struct.std::__1::__cxx_atomic_base_impl.44" = type { i128 }
%"struct.std::__1::atomic.45" = type { %"struct.std::__1::__atomic_base.46" }
%"struct.std::__1::__atomic_base.46" = type { %"struct.std::__1::__atomic_base.47" }
%"struct.std::__1::__atomic_base.47" = type { %"struct.std::__1::__cxx_atomic_impl.48" }
%"struct.std::__1::__cxx_atomic_impl.48" = type { %"struct.std::__1::__cxx_atomic_base_impl.49" }
%"struct.std::__1::__cxx_atomic_base_impl.49" = type { i128 }

; Function Attrs: nofree norecurse nounwind
define zeroext i1 @_Z22atomic_load_relaxed_i1RNSt3__16atomicIbEE(%"struct.std::__1::atomic"* nocapture nonnull readonly align 1 dereferenceable(1) %0) {
; CHECK-LABEL: _Z22atomic_load_relaxed_i1RNSt3__16atomicIbEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    and %s0, 1, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic", %"struct.std::__1::atomic"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i8, i8* %2 monotonic, align 1
  %4 = and i8 %3, 1
  %5 = icmp ne i8 %4, 0
  ret i1 %5
}

; Function Attrs: nofree norecurse nounwind
define signext i8 @_Z22atomic_load_relaxed_i8RNSt3__16atomicIcEE(%"struct.std::__1::atomic.0"* nocapture nonnull readonly align 1 dereferenceable(1) %0) {
; CHECK-LABEL: _Z22atomic_load_relaxed_i8RNSt3__16atomicIcEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.sx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i8, i8* %2 monotonic, align 1
  ret i8 %3
}

; Function Attrs: nofree norecurse nounwind
define zeroext i8 @_Z22atomic_load_relaxed_u8RNSt3__16atomicIhEE(%"struct.std::__1::atomic.5"* nocapture nonnull readonly align 1 dereferenceable(1) %0) {
; CHECK-LABEL: _Z22atomic_load_relaxed_u8RNSt3__16atomicIhEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i8, i8* %2 monotonic, align 1
  ret i8 %3
}

; Function Attrs: nofree norecurse nounwind
define signext i16 @_Z23atomic_load_relaxed_i16RNSt3__16atomicIsEE(%"struct.std::__1::atomic.10"* nocapture nonnull readonly align 2 dereferenceable(2) %0) {
; CHECK-LABEL: _Z23atomic_load_relaxed_i16RNSt3__16atomicIsEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld2b.sx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i16, i16* %2 monotonic, align 2
  ret i16 %3
}

; Function Attrs: nofree norecurse nounwind
define zeroext i16 @_Z23atomic_load_relaxed_u16RNSt3__16atomicItEE(%"struct.std::__1::atomic.15"* nocapture nonnull readonly align 2 dereferenceable(2) %0) {
; CHECK-LABEL: _Z23atomic_load_relaxed_u16RNSt3__16atomicItEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i16, i16* %2 monotonic, align 2
  ret i16 %3
}

; Function Attrs: nofree norecurse nounwind
define signext i32 @_Z23atomic_load_relaxed_i32RNSt3__16atomicIiEE(%"struct.std::__1::atomic.20"* nocapture nonnull readonly align 4 dereferenceable(4) %0) {
; CHECK-LABEL: _Z23atomic_load_relaxed_i32RNSt3__16atomicIiEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i32, i32* %2 monotonic, align 4
  ret i32 %3
}

; Function Attrs: nofree norecurse nounwind
define zeroext i32 @_Z23atomic_load_relaxed_u32RNSt3__16atomicIjEE(%"struct.std::__1::atomic.25"* nocapture nonnull readonly align 4 dereferenceable(4) %0) {
; CHECK-LABEL: _Z23atomic_load_relaxed_u32RNSt3__16atomicIjEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.zx %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i32, i32* %2 monotonic, align 4
  ret i32 %3
}

; Function Attrs: nofree norecurse nounwind
define i64 @_Z23atomic_load_relaxed_i64RNSt3__16atomicIlEE(%"struct.std::__1::atomic.30"* nocapture nonnull readonly align 8 dereferenceable(8) %0) {
; CHECK-LABEL: _Z23atomic_load_relaxed_i64RNSt3__16atomicIlEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i64, i64* %2 monotonic, align 8
  ret i64 %3
}

; Function Attrs: nofree norecurse nounwind
define i64 @_Z23atomic_load_relaxed_u64RNSt3__16atomicImEE(%"struct.std::__1::atomic.35"* nocapture nonnull readonly align 8 dereferenceable(8) %0) {
; CHECK-LABEL: _Z23atomic_load_relaxed_u64RNSt3__16atomicImEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i64, i64* %2 monotonic, align 8
  ret i64 %3
}

; Function Attrs: nounwind
define i128 @_Z24atomic_load_relaxed_i128RNSt3__16atomicInEE(%"struct.std::__1::atomic.40"* nonnull align 16 dereferenceable(16) %0) {
; CHECK-LABEL: _Z24atomic_load_relaxed_i128RNSt3__16atomicInEE:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, __atomic_load@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_load@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    ld %s1, 248(, %s11)
; CHECK-NEXT:    ld %s0, 240(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = alloca i128, align 16
  %3 = bitcast i128* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3)
  %4 = bitcast %"struct.std::__1::atomic.40"* %0 to i8*
  call void @__atomic_load(i64 16, i8* nonnull %4, i8* nonnull %3, i32 signext 0)
  %5 = load i128, i128* %2, align 16, !tbaa !2
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3)
  ret i128 %5
}

; Function Attrs: nounwind
define i128 @_Z24atomic_load_relaxed_u128RNSt3__16atomicIoEE(%"struct.std::__1::atomic.45"* nonnull align 16 dereferenceable(16) %0) {
; CHECK-LABEL: _Z24atomic_load_relaxed_u128RNSt3__16atomicIoEE:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, __atomic_load@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_load@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    ld %s1, 248(, %s11)
; CHECK-NEXT:    ld %s0, 240(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = alloca i128, align 16
  %3 = bitcast i128* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3)
  %4 = bitcast %"struct.std::__1::atomic.45"* %0 to i8*
  call void @__atomic_load(i64 16, i8* nonnull %4, i8* nonnull %3, i32 signext 0)
  %5 = load i128, i128* %2, align 16, !tbaa !2
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3)
  ret i128 %5
}

; Function Attrs: nofree norecurse nounwind
define zeroext i1 @_Z22atomic_load_acquire_i1RNSt3__16atomicIbEE(%"struct.std::__1::atomic"* nocapture nonnull readonly align 1 dereferenceable(1) %0) {
; CHECK-LABEL: _Z22atomic_load_acquire_i1RNSt3__16atomicIbEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    and %s0, 1, %s0
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic", %"struct.std::__1::atomic"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i8, i8* %2 acquire, align 1
  %4 = and i8 %3, 1
  %5 = icmp ne i8 %4, 0
  ret i1 %5
}

; Function Attrs: nofree norecurse nounwind
define signext i8 @_Z22atomic_load_acquire_i8RNSt3__16atomicIcEE(%"struct.std::__1::atomic.0"* nocapture nonnull readonly align 1 dereferenceable(1) %0) {
; CHECK-LABEL: _Z22atomic_load_acquire_i8RNSt3__16atomicIcEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.sx %s0, (, %s0)
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i8, i8* %2 acquire, align 1
  ret i8 %3
}

; Function Attrs: nofree norecurse nounwind
define zeroext i8 @_Z22atomic_load_acquire_u8RNSt3__16atomicIhEE(%"struct.std::__1::atomic.5"* nocapture nonnull readonly align 1 dereferenceable(1) %0) {
; CHECK-LABEL: _Z22atomic_load_acquire_u8RNSt3__16atomicIhEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i8, i8* %2 acquire, align 1
  ret i8 %3
}

; Function Attrs: nofree norecurse nounwind
define signext i16 @_Z23atomic_load_acquire_i16RNSt3__16atomicIsEE(%"struct.std::__1::atomic.10"* nocapture nonnull readonly align 2 dereferenceable(2) %0) {
; CHECK-LABEL: _Z23atomic_load_acquire_i16RNSt3__16atomicIsEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld2b.sx %s0, (, %s0)
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i16, i16* %2 acquire, align 2
  ret i16 %3
}

; Function Attrs: nofree norecurse nounwind
define zeroext i16 @_Z23atomic_load_acquire_u16RNSt3__16atomicItEE(%"struct.std::__1::atomic.15"* nocapture nonnull readonly align 2 dereferenceable(2) %0) {
; CHECK-LABEL: _Z23atomic_load_acquire_u16RNSt3__16atomicItEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i16, i16* %2 acquire, align 2
  ret i16 %3
}

; Function Attrs: nofree norecurse nounwind
define signext i32 @_Z23atomic_load_acquire_i32RNSt3__16atomicIiEE(%"struct.std::__1::atomic.20"* nocapture nonnull readonly align 4 dereferenceable(4) %0) {
; CHECK-LABEL: _Z23atomic_load_acquire_i32RNSt3__16atomicIiEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s0, (, %s0)
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i32, i32* %2 acquire, align 4
  ret i32 %3
}

; Function Attrs: nofree norecurse nounwind
define zeroext i32 @_Z23atomic_load_acquire_u32RNSt3__16atomicIjEE(%"struct.std::__1::atomic.25"* nocapture nonnull readonly align 4 dereferenceable(4) %0) {
; CHECK-LABEL: _Z23atomic_load_acquire_u32RNSt3__16atomicIjEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.zx %s0, (, %s0)
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i32, i32* %2 acquire, align 4
  ret i32 %3
}

; Function Attrs: nofree norecurse nounwind
define i64 @_Z23atomic_load_acquire_i64RNSt3__16atomicIlEE(%"struct.std::__1::atomic.30"* nocapture nonnull readonly align 8 dereferenceable(8) %0) {
; CHECK-LABEL: _Z23atomic_load_acquire_i64RNSt3__16atomicIlEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i64, i64* %2 acquire, align 8
  ret i64 %3
}

; Function Attrs: nofree norecurse nounwind
define i64 @_Z23atomic_load_acquire_u64RNSt3__16atomicImEE(%"struct.std::__1::atomic.35"* nocapture nonnull readonly align 8 dereferenceable(8) %0) {
; CHECK-LABEL: _Z23atomic_load_acquire_u64RNSt3__16atomicImEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i64, i64* %2 acquire, align 8
  ret i64 %3
}

; Function Attrs: nounwind
define i128 @_Z24atomic_load_acquire_i128RNSt3__16atomicInEE(%"struct.std::__1::atomic.40"* nonnull align 16 dereferenceable(16) %0) {
; CHECK-LABEL: _Z24atomic_load_acquire_i128RNSt3__16atomicInEE:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, __atomic_load@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_load@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    ld %s1, 248(, %s11)
; CHECK-NEXT:    ld %s0, 240(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = alloca i128, align 16
  %3 = bitcast i128* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3)
  %4 = bitcast %"struct.std::__1::atomic.40"* %0 to i8*
  call void @__atomic_load(i64 16, i8* nonnull %4, i8* nonnull %3, i32 signext 2)
  %5 = load i128, i128* %2, align 16, !tbaa !2
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3)
  ret i128 %5
}

; Function Attrs: nounwind
define i128 @_Z24atomic_load_acquire_u128RNSt3__16atomicIoEE(%"struct.std::__1::atomic.45"* nonnull align 16 dereferenceable(16) %0) {
; CHECK-LABEL: _Z24atomic_load_acquire_u128RNSt3__16atomicIoEE:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, __atomic_load@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_load@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 2, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    ld %s1, 248(, %s11)
; CHECK-NEXT:    ld %s0, 240(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = alloca i128, align 16
  %3 = bitcast i128* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3)
  %4 = bitcast %"struct.std::__1::atomic.45"* %0 to i8*
  call void @__atomic_load(i64 16, i8* nonnull %4, i8* nonnull %3, i32 signext 2)
  %5 = load i128, i128* %2, align 16, !tbaa !2
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3)
  ret i128 %5
}

; Function Attrs: nofree norecurse nounwind
define zeroext i1 @_Z22atomic_load_seq_cst_i1RNSt3__16atomicIbEE(%"struct.std::__1::atomic"* nocapture nonnull readonly align 1 dereferenceable(1) %0) {
; CHECK-LABEL: _Z22atomic_load_seq_cst_i1RNSt3__16atomicIbEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    and %s0, 1, %s0
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic", %"struct.std::__1::atomic"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i8, i8* %2 seq_cst, align 1
  %4 = and i8 %3, 1
  %5 = icmp ne i8 %4, 0
  ret i1 %5
}

; Function Attrs: nofree norecurse nounwind
define signext i8 @_Z22atomic_load_seq_cst_i8RNSt3__16atomicIcEE(%"struct.std::__1::atomic.0"* nocapture nonnull readonly align 1 dereferenceable(1) %0) {
; CHECK-LABEL: _Z22atomic_load_seq_cst_i8RNSt3__16atomicIcEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.sx %s0, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i8, i8* %2 seq_cst, align 1
  ret i8 %3
}

; Function Attrs: nofree norecurse nounwind
define zeroext i8 @_Z22atomic_load_seq_cst_u8RNSt3__16atomicIhEE(%"struct.std::__1::atomic.5"* nocapture nonnull readonly align 1 dereferenceable(1) %0) {
; CHECK-LABEL: _Z22atomic_load_seq_cst_u8RNSt3__16atomicIhEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld1b.zx %s0, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i8, i8* %2 seq_cst, align 1
  ret i8 %3
}

; Function Attrs: nofree norecurse nounwind
define signext i16 @_Z23atomic_load_seq_cst_i16RNSt3__16atomicIsEE(%"struct.std::__1::atomic.10"* nocapture nonnull readonly align 2 dereferenceable(2) %0) {
; CHECK-LABEL: _Z23atomic_load_seq_cst_i16RNSt3__16atomicIsEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld2b.sx %s0, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i16, i16* %2 seq_cst, align 2
  ret i16 %3
}

; Function Attrs: nofree norecurse nounwind
define zeroext i16 @_Z23atomic_load_seq_cst_u16RNSt3__16atomicItEE(%"struct.std::__1::atomic.15"* nocapture nonnull readonly align 2 dereferenceable(2) %0) {
; CHECK-LABEL: _Z23atomic_load_seq_cst_u16RNSt3__16atomicItEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i16, i16* %2 seq_cst, align 2
  ret i16 %3
}

; Function Attrs: nofree norecurse nounwind
define signext i32 @_Z23atomic_load_seq_cst_i32RNSt3__16atomicIiEE(%"struct.std::__1::atomic.20"* nocapture nonnull readonly align 4 dereferenceable(4) %0) {
; CHECK-LABEL: _Z23atomic_load_seq_cst_i32RNSt3__16atomicIiEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.sx %s0, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i32, i32* %2 seq_cst, align 4
  ret i32 %3
}

; Function Attrs: nofree norecurse nounwind
define zeroext i32 @_Z23atomic_load_seq_cst_u32RNSt3__16atomicIjEE(%"struct.std::__1::atomic.25"* nocapture nonnull readonly align 4 dereferenceable(4) %0) {
; CHECK-LABEL: _Z23atomic_load_seq_cst_u32RNSt3__16atomicIjEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldl.zx %s0, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i32, i32* %2 seq_cst, align 4
  ret i32 %3
}

; Function Attrs: nofree norecurse nounwind
define i64 @_Z23atomic_load_seq_cst_i64RNSt3__16atomicIlEE(%"struct.std::__1::atomic.30"* nocapture nonnull readonly align 8 dereferenceable(8) %0) {
; CHECK-LABEL: _Z23atomic_load_seq_cst_i64RNSt3__16atomicIlEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i64, i64* %2 seq_cst, align 8
  ret i64 %3
}

; Function Attrs: nofree norecurse nounwind
define i64 @_Z23atomic_load_seq_cst_u64RNSt3__16atomicImEE(%"struct.std::__1::atomic.35"* nocapture nonnull readonly align 8 dereferenceable(8) %0) {
; CHECK-LABEL: _Z23atomic_load_seq_cst_u64RNSt3__16atomicImEE:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ld %s0, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = getelementptr inbounds %"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  %3 = load atomic i64, i64* %2 seq_cst, align 8
  ret i64 %3
}

; Function Attrs: nounwind
define i128 @_Z24atomic_load_seq_cst_i128RNSt3__16atomicInEE(%"struct.std::__1::atomic.40"* nonnull align 16 dereferenceable(16) %0) {
; CHECK-LABEL: _Z24atomic_load_seq_cst_i128RNSt3__16atomicInEE:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, __atomic_load@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_load@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 5, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    ld %s1, 248(, %s11)
; CHECK-NEXT:    ld %s0, 240(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = alloca i128, align 16
  %3 = bitcast i128* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3)
  %4 = bitcast %"struct.std::__1::atomic.40"* %0 to i8*
  call void @__atomic_load(i64 16, i8* nonnull %4, i8* nonnull %3, i32 signext 5)
  %5 = load i128, i128* %2, align 16, !tbaa !2
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3)
  ret i128 %5
}

; Function Attrs: nounwind
define i128 @_Z24atomic_load_seq_cst_u128RNSt3__16atomicIoEE(%"struct.std::__1::atomic.45"* nonnull align 16 dereferenceable(16) %0) {
; CHECK-LABEL: _Z24atomic_load_seq_cst_u128RNSt3__16atomicIoEE:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s1, 0, %s0
; CHECK-NEXT:    lea %s0, __atomic_load@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_load@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 5, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    ld %s1, 248(, %s11)
; CHECK-NEXT:    ld %s0, 240(, %s11)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = alloca i128, align 16
  %3 = bitcast i128* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3)
  %4 = bitcast %"struct.std::__1::atomic.45"* %0 to i8*
  call void @__atomic_load(i64 16, i8* nonnull %4, i8* nonnull %3, i32 signext 5)
  %5 = load i128, i128* %2, align 16, !tbaa !2
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3)
  ret i128 %5
}

; Function Attrs: nofree nounwind willreturn
declare void @__atomic_load(i64, i8*, i8*, i32)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

!2 = !{!3, !3, i64 0}
!3 = !{!"__int128", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
