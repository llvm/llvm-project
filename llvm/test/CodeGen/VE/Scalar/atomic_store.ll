; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test atomic store for all types and all memory order
;;;
;;; Note:
;;;   We test i1/i8/i16/i32/i64/i128/u8/u16/u32/u64/u128.
;;;   We test relaxed, release, and seq_cst.
;;;   We test an object, a stack object, and a global variable.

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

@gv_i1 = global %"struct.std::__1::atomic" zeroinitializer, align 4
@gv_i8 = global %"struct.std::__1::atomic.0" zeroinitializer, align 4
@gv_u8 = global %"struct.std::__1::atomic.5" zeroinitializer, align 4
@gv_i16 = global %"struct.std::__1::atomic.10" zeroinitializer, align 4
@gv_u16 = global %"struct.std::__1::atomic.15" zeroinitializer, align 4
@gv_i32 = global %"struct.std::__1::atomic.20" zeroinitializer, align 4
@gv_u32 = global %"struct.std::__1::atomic.25" zeroinitializer, align 4
@gv_i64 = global %"struct.std::__1::atomic.30" zeroinitializer, align 8
@gv_u64 = global %"struct.std::__1::atomic.35" zeroinitializer, align 8
@gv_i128 = global %"struct.std::__1::atomic.40" zeroinitializer, align 16
@gv_u128 = global %"struct.std::__1::atomic.45" zeroinitializer, align 16

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_relaxed_i1RNSt3__16atomicIbEEb(%"struct.std::__1::atomic"* nocapture nonnull align 1 dereferenceable(1) %0, i1 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_relaxed_i1RNSt3__16atomicIbEEb:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i1 %1 to i8
  %4 = getelementptr inbounds %"struct.std::__1::atomic", %"struct.std::__1::atomic"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  store atomic i8 %3, i8* %4 monotonic, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_relaxed_i8RNSt3__16atomicIcEEc(%"struct.std::__1::atomic.0"* nocapture nonnull align 1 dereferenceable(1) %0, i8 signext %1) {
; CHECK-LABEL: _Z23atomic_store_relaxed_i8RNSt3__16atomicIcEEc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i8 %1, i8* %3 monotonic, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_relaxed_u8RNSt3__16atomicIhEEh(%"struct.std::__1::atomic.5"* nocapture nonnull align 1 dereferenceable(1) %0, i8 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_relaxed_u8RNSt3__16atomicIhEEh:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i8 %1, i8* %3 monotonic, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_i16RNSt3__16atomicIsEEs(%"struct.std::__1::atomic.10"* nocapture nonnull align 2 dereferenceable(2) %0, i16 signext %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_i16RNSt3__16atomicIsEEs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i16 %1, i16* %3 monotonic, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_u16RNSt3__16atomicItEEt(%"struct.std::__1::atomic.15"* nocapture nonnull align 2 dereferenceable(2) %0, i16 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_u16RNSt3__16atomicItEEt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i16 %1, i16* %3 monotonic, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_i32RNSt3__16atomicIiEEi(%"struct.std::__1::atomic.20"* nocapture nonnull align 4 dereferenceable(4) %0, i32 signext %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_i32RNSt3__16atomicIiEEi:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i32 %1, i32* %3 monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_u32RNSt3__16atomicIjEEj(%"struct.std::__1::atomic.25"* nocapture nonnull align 4 dereferenceable(4) %0, i32 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_u32RNSt3__16atomicIjEEj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i32 %1, i32* %3 monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_i64RNSt3__16atomicIlEEl(%"struct.std::__1::atomic.30"* nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_i64RNSt3__16atomicIlEEl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i64 %1, i64* %3 monotonic, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_u64RNSt3__16atomicImEEm(%"struct.std::__1::atomic.35"* nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_u64RNSt3__16atomicImEEm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i64 %1, i64* %3 monotonic, align 8
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_relaxed_i128RNSt3__16atomicInEEn(%"struct.std::__1::atomic.40"* nonnull align 16 dereferenceable(16) %0, i128 %1) {
; CHECK-LABEL: _Z25atomic_store_relaxed_i128RNSt3__16atomicInEEn:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s4, 0, %s0
; CHECK-NEXT:    st %s2, 248(, %s11)
; CHECK-NEXT:    st %s1, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_store@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_store@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    or %s1, 0, %s4
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca i128, align 16
  %4 = bitcast i128* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4)
  store i128 %1, i128* %3, align 16, !tbaa !2
  %5 = bitcast %"struct.std::__1::atomic.40"* %0 to i8*
  call void @__atomic_store(i64 16, i8* nonnull %5, i8* nonnull %4, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_relaxed_u128RNSt3__16atomicIoEEo(%"struct.std::__1::atomic.45"* nonnull align 16 dereferenceable(16) %0, i128 %1) {
; CHECK-LABEL: _Z25atomic_store_relaxed_u128RNSt3__16atomicIoEEo:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s4, 0, %s0
; CHECK-NEXT:    st %s2, 248(, %s11)
; CHECK-NEXT:    st %s1, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_store@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_store@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    or %s1, 0, %s4
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca i128, align 16
  %4 = bitcast i128* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4)
  store i128 %1, i128* %3, align 16, !tbaa !2
  %5 = bitcast %"struct.std::__1::atomic.45"* %0 to i8*
  call void @__atomic_store(i64 16, i8* nonnull %5, i8* nonnull %4, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4)
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_release_i1RNSt3__16atomicIbEEb(%"struct.std::__1::atomic"* nocapture nonnull align 1 dereferenceable(1) %0, i1 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_release_i1RNSt3__16atomicIbEEb:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i1 %1 to i8
  %4 = getelementptr inbounds %"struct.std::__1::atomic", %"struct.std::__1::atomic"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  store atomic i8 %3, i8* %4 release, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_release_i8RNSt3__16atomicIcEEc(%"struct.std::__1::atomic.0"* nocapture nonnull align 1 dereferenceable(1) %0, i8 signext %1) {
; CHECK-LABEL: _Z23atomic_store_release_i8RNSt3__16atomicIcEEc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i8 %1, i8* %3 release, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_release_u8RNSt3__16atomicIhEEh(%"struct.std::__1::atomic.5"* nocapture nonnull align 1 dereferenceable(1) %0, i8 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_release_u8RNSt3__16atomicIhEEh:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i8 %1, i8* %3 release, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_i16RNSt3__16atomicIsEEs(%"struct.std::__1::atomic.10"* nocapture nonnull align 2 dereferenceable(2) %0, i16 signext %1) {
; CHECK-LABEL: _Z24atomic_store_release_i16RNSt3__16atomicIsEEs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i16 %1, i16* %3 release, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_u16RNSt3__16atomicItEEt(%"struct.std::__1::atomic.15"* nocapture nonnull align 2 dereferenceable(2) %0, i16 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_release_u16RNSt3__16atomicItEEt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i16 %1, i16* %3 release, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_i32RNSt3__16atomicIiEEi(%"struct.std::__1::atomic.20"* nocapture nonnull align 4 dereferenceable(4) %0, i32 signext %1) {
; CHECK-LABEL: _Z24atomic_store_release_i32RNSt3__16atomicIiEEi:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i32 %1, i32* %3 release, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_u32RNSt3__16atomicIjEEj(%"struct.std::__1::atomic.25"* nocapture nonnull align 4 dereferenceable(4) %0, i32 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_release_u32RNSt3__16atomicIjEEj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i32 %1, i32* %3 release, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_i64RNSt3__16atomicIlEEl(%"struct.std::__1::atomic.30"* nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_release_i64RNSt3__16atomicIlEEl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i64 %1, i64* %3 release, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_u64RNSt3__16atomicImEEm(%"struct.std::__1::atomic.35"* nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_release_u64RNSt3__16atomicImEEm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i64 %1, i64* %3 release, align 8
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_release_i128RNSt3__16atomicInEEn(%"struct.std::__1::atomic.40"* nonnull align 16 dereferenceable(16) %0, i128 %1) {
; CHECK-LABEL: _Z25atomic_store_release_i128RNSt3__16atomicInEEn:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s4, 0, %s0
; CHECK-NEXT:    st %s2, 248(, %s11)
; CHECK-NEXT:    st %s1, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_store@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_store@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 3, (0)1
; CHECK-NEXT:    or %s1, 0, %s4
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca i128, align 16
  %4 = bitcast i128* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4)
  store i128 %1, i128* %3, align 16, !tbaa !2
  %5 = bitcast %"struct.std::__1::atomic.40"* %0 to i8*
  call void @__atomic_store(i64 16, i8* nonnull %5, i8* nonnull %4, i32 signext 3)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_release_u128RNSt3__16atomicIoEEo(%"struct.std::__1::atomic.45"* nonnull align 16 dereferenceable(16) %0, i128 %1) {
; CHECK-LABEL: _Z25atomic_store_release_u128RNSt3__16atomicIoEEo:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s4, 0, %s0
; CHECK-NEXT:    st %s2, 248(, %s11)
; CHECK-NEXT:    st %s1, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_store@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_store@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 3, (0)1
; CHECK-NEXT:    or %s1, 0, %s4
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca i128, align 16
  %4 = bitcast i128* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4)
  store i128 %1, i128* %3, align 16, !tbaa !2
  %5 = bitcast %"struct.std::__1::atomic.45"* %0 to i8*
  call void @__atomic_store(i64 16, i8* nonnull %5, i8* nonnull %4, i32 signext 3)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4)
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_seq_cst_i1RNSt3__16atomicIbEEb(%"struct.std::__1::atomic"* nocapture nonnull align 1 dereferenceable(1) %0, i1 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_seq_cst_i1RNSt3__16atomicIbEEb:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i1 %1 to i8
  %4 = getelementptr inbounds %"struct.std::__1::atomic", %"struct.std::__1::atomic"* %0, i64 0, i32 0, i32 0, i32 0, i32 0
  store atomic i8 %3, i8* %4 seq_cst, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_seq_cst_i8RNSt3__16atomicIcEEc(%"struct.std::__1::atomic.0"* nocapture nonnull align 1 dereferenceable(1) %0, i8 signext %1) {
; CHECK-LABEL: _Z23atomic_store_seq_cst_i8RNSt3__16atomicIcEEc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i8 %1, i8* %3 seq_cst, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_seq_cst_u8RNSt3__16atomicIhEEh(%"struct.std::__1::atomic.5"* nocapture nonnull align 1 dereferenceable(1) %0, i8 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_seq_cst_u8RNSt3__16atomicIhEEh:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i8 %1, i8* %3 seq_cst, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_i16RNSt3__16atomicIsEEs(%"struct.std::__1::atomic.10"* nocapture nonnull align 2 dereferenceable(2) %0, i16 signext %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_i16RNSt3__16atomicIsEEs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i16 %1, i16* %3 seq_cst, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_u16RNSt3__16atomicItEEt(%"struct.std::__1::atomic.15"* nocapture nonnull align 2 dereferenceable(2) %0, i16 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_u16RNSt3__16atomicItEEt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i16 %1, i16* %3 seq_cst, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_i32RNSt3__16atomicIiEEi(%"struct.std::__1::atomic.20"* nocapture nonnull align 4 dereferenceable(4) %0, i32 signext %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_i32RNSt3__16atomicIiEEi:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i32 %1, i32* %3 seq_cst, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_u32RNSt3__16atomicIjEEj(%"struct.std::__1::atomic.25"* nocapture nonnull align 4 dereferenceable(4) %0, i32 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_u32RNSt3__16atomicIjEEj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i32 %1, i32* %3 seq_cst, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_i64RNSt3__16atomicIlEEl(%"struct.std::__1::atomic.30"* nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_i64RNSt3__16atomicIlEEl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i64 %1, i64* %3 seq_cst, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_u64RNSt3__16atomicImEEm(%"struct.std::__1::atomic.35"* nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_u64RNSt3__16atomicImEEm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = getelementptr inbounds %"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* %0, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0
  store atomic i64 %1, i64* %3 seq_cst, align 8
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_seq_cst_i128RNSt3__16atomicInEEn(%"struct.std::__1::atomic.40"* nonnull align 16 dereferenceable(16) %0, i128 %1) {
; CHECK-LABEL: _Z25atomic_store_seq_cst_i128RNSt3__16atomicInEEn:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s4, 0, %s0
; CHECK-NEXT:    st %s2, 248(, %s11)
; CHECK-NEXT:    st %s1, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_store@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_store@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 5, (0)1
; CHECK-NEXT:    or %s1, 0, %s4
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca i128, align 16
  %4 = bitcast i128* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4)
  store i128 %1, i128* %3, align 16, !tbaa !2
  %5 = bitcast %"struct.std::__1::atomic.40"* %0 to i8*
  call void @__atomic_store(i64 16, i8* nonnull %5, i8* nonnull %4, i32 signext 5)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_seq_cst_u128RNSt3__16atomicIoEEo(%"struct.std::__1::atomic.45"* nonnull align 16 dereferenceable(16) %0, i128 %1) {
; CHECK-LABEL: _Z25atomic_store_seq_cst_u128RNSt3__16atomicIoEEo:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s4, 0, %s0
; CHECK-NEXT:    st %s2, 248(, %s11)
; CHECK-NEXT:    st %s1, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_store@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_store@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 5, (0)1
; CHECK-NEXT:    or %s1, 0, %s4
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = alloca i128, align 16
  %4 = bitcast i128* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4)
  store i128 %1, i128* %3, align 16, !tbaa !2
  %5 = bitcast %"struct.std::__1::atomic.45"* %0 to i8*
  call void @__atomic_store(i64 16, i8* nonnull %5, i8* nonnull %4, i32 signext 5)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z26atomic_load_relaxed_stk_i1b(i1 zeroext %0) {
; CHECK-LABEL: _Z26atomic_load_relaxed_stk_i1b:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st1b %s0, 15(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = alloca i8, align 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %2)
  %3 = zext i1 %0 to i8
  store atomic volatile i8 %3, i8* %2 monotonic, align 1
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %2)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: nofree nounwind mustprogress
define void @_Z26atomic_load_relaxed_stk_i8c(i8 signext %0) {
; CHECK-LABEL: _Z26atomic_load_relaxed_stk_i8c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st1b %s0, 15(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = alloca i8, align 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %2)
  store atomic volatile i8 %0, i8* %2 monotonic, align 1
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %2)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z26atomic_load_relaxed_stk_u8h(i8 zeroext %0) {
; CHECK-LABEL: _Z26atomic_load_relaxed_stk_u8h:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st1b %s0, 15(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = alloca i8, align 1
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %2)
  store atomic volatile i8 %0, i8* %2 monotonic, align 1
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %2)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z27atomic_load_relaxed_stk_i16s(i16 signext %0) {
; CHECK-LABEL: _Z27atomic_load_relaxed_stk_i16s:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st2b %s0, 14(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = alloca i16, align 2
  %3 = bitcast i16* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %3)
  store atomic volatile i16 %0, i16* %2 monotonic, align 2
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z27atomic_load_relaxed_stk_u16t(i16 zeroext %0) {
; CHECK-LABEL: _Z27atomic_load_relaxed_stk_u16t:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st2b %s0, 14(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = alloca i16, align 2
  %3 = bitcast i16* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 2, i8* nonnull %3)
  store atomic volatile i16 %0, i16* %2 monotonic, align 2
  call void @llvm.lifetime.end.p0i8(i64 2, i8* nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z27atomic_load_relaxed_stk_i32i(i32 signext %0) {
; CHECK-LABEL: _Z27atomic_load_relaxed_stk_i32i:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    stl %s0, 12(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = alloca i32, align 4
  %3 = bitcast i32* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %3)
  store atomic volatile i32 %0, i32* %2 monotonic, align 4
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z27atomic_load_relaxed_stk_u32j(i32 zeroext %0) {
; CHECK-LABEL: _Z27atomic_load_relaxed_stk_u32j:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    stl %s0, 12(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = alloca i32, align 4
  %3 = bitcast i32* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %3)
  store atomic volatile i32 %0, i32* %2 monotonic, align 4
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z27atomic_load_relaxed_stk_i64l(i64 %0) {
; CHECK-LABEL: _Z27atomic_load_relaxed_stk_i64l:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s0, 8(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = alloca i64, align 8
  %3 = bitcast i64* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3)
  store atomic volatile i64 %0, i64* %2 monotonic, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z27atomic_load_relaxed_stk_u64m(i64 %0) {
; CHECK-LABEL: _Z27atomic_load_relaxed_stk_u64m:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s0, 8(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = alloca i64, align 8
  %3 = bitcast i64* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %3)
  store atomic volatile i64 %0, i64* %2 monotonic, align 8
  call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z28atomic_load_relaxed_stk_i128n(i128 %0) {
; CHECK-LABEL: _Z28atomic_load_relaxed_stk_i128n:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s1, 264(, %s11)
; CHECK-NEXT:    st %s0, 256(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_store@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_store@hi(, %s0)
; CHECK-NEXT:    lea %s1, 240(, %s11)
; CHECK-NEXT:    lea %s2, 256(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = alloca i128, align 16
  %3 = alloca %"struct.std::__1::atomic.40", align 16
  %4 = bitcast %"struct.std::__1::atomic.40"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4)
  %5 = bitcast i128* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5)
  store i128 %0, i128* %2, align 16, !tbaa !2
  call void @__atomic_store(i64 16, i8* nonnull %4, i8* nonnull %5, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z28atomic_load_relaxed_stk_u128o(i128 %0) {
; CHECK-LABEL: _Z28atomic_load_relaxed_stk_u128o:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s1, 264(, %s11)
; CHECK-NEXT:    st %s0, 256(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_store@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_store@hi(, %s0)
; CHECK-NEXT:    lea %s1, 240(, %s11)
; CHECK-NEXT:    lea %s2, 256(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = alloca i128, align 16
  %3 = alloca %"struct.std::__1::atomic.45", align 16
  %4 = bitcast %"struct.std::__1::atomic.45"* %3 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %4)
  %5 = bitcast i128* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %5)
  store i128 %0, i128* %2, align 16, !tbaa !2
  call void @__atomic_store(i64 16, i8* nonnull %4, i8* nonnull %5, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %5)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %4)
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z25atomic_load_relaxed_gv_i1b(i1 zeroext %0) {
; CHECK-LABEL: _Z25atomic_load_relaxed_gv_i1b:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, gv_i1@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_i1@hi(, %s1)
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = zext i1 %0 to i8
  store atomic i8 %2, i8* getelementptr inbounds (%"struct.std::__1::atomic", %"struct.std::__1::atomic"* @gv_i1, i64 0, i32 0, i32 0, i32 0, i32 0) monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z25atomic_load_relaxed_gv_i8c(i8 signext %0) {
; CHECK-LABEL: _Z25atomic_load_relaxed_gv_i8c:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, gv_i8@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_i8@hi(, %s1)
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i8 %0, i8* getelementptr inbounds (%"struct.std::__1::atomic.0", %"struct.std::__1::atomic.0"* @gv_i8, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0) monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z25atomic_load_relaxed_gv_u8h(i8 zeroext %0) {
; CHECK-LABEL: _Z25atomic_load_relaxed_gv_u8h:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, gv_u8@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_u8@hi(, %s1)
; CHECK-NEXT:    st1b %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i8 %0, i8* getelementptr inbounds (%"struct.std::__1::atomic.5", %"struct.std::__1::atomic.5"* @gv_u8, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0) monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z26atomic_load_relaxed_gv_i16s(i16 signext %0) {
; CHECK-LABEL: _Z26atomic_load_relaxed_gv_i16s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, gv_i16@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_i16@hi(, %s1)
; CHECK-NEXT:    st2b %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i16 %0, i16* getelementptr inbounds (%"struct.std::__1::atomic.10", %"struct.std::__1::atomic.10"* @gv_i16, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0) monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z26atomic_load_relaxed_gv_u16t(i16 zeroext %0) {
; CHECK-LABEL: _Z26atomic_load_relaxed_gv_u16t:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, gv_u16@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_u16@hi(, %s1)
; CHECK-NEXT:    st2b %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i16 %0, i16* getelementptr inbounds (%"struct.std::__1::atomic.15", %"struct.std::__1::atomic.15"* @gv_u16, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0) monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z26atomic_load_relaxed_gv_i32i(i32 signext %0) {
; CHECK-LABEL: _Z26atomic_load_relaxed_gv_i32i:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, gv_i32@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_i32@hi(, %s1)
; CHECK-NEXT:    stl %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i32 %0, i32* getelementptr inbounds (%"struct.std::__1::atomic.20", %"struct.std::__1::atomic.20"* @gv_i32, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0) monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z26atomic_load_relaxed_gv_u32j(i32 zeroext %0) {
; CHECK-LABEL: _Z26atomic_load_relaxed_gv_u32j:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, gv_u32@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_u32@hi(, %s1)
; CHECK-NEXT:    stl %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i32 %0, i32* getelementptr inbounds (%"struct.std::__1::atomic.25", %"struct.std::__1::atomic.25"* @gv_u32, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0) monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z26atomic_load_relaxed_gv_i64l(i64 %0) {
; CHECK-LABEL: _Z26atomic_load_relaxed_gv_i64l:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, gv_i64@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_i64@hi(, %s1)
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i64 %0, i64* getelementptr inbounds (%"struct.std::__1::atomic.30", %"struct.std::__1::atomic.30"* @gv_i64, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0) monotonic, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z26atomic_load_relaxed_gv_u64m(i64 %0) {
; CHECK-LABEL: _Z26atomic_load_relaxed_gv_u64m:
; CHECK:       # %bb.0:
; CHECK-NEXT:    lea %s1, gv_u64@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_u64@hi(, %s1)
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i64 %0, i64* getelementptr inbounds (%"struct.std::__1::atomic.35", %"struct.std::__1::atomic.35"* @gv_u64, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0) monotonic, align 8
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z27atomic_load_relaxed_gv_i128n(i128 %0) {
; CHECK-LABEL: _Z27atomic_load_relaxed_gv_i128n:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s1, 248(, %s11)
; CHECK-NEXT:    st %s0, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_store@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_store@hi(, %s0)
; CHECK-NEXT:    lea %s0, gv_i128@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_i128@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = alloca i128, align 16
  %3 = bitcast i128* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3)
  store i128 %0, i128* %2, align 16, !tbaa !2
  call void @__atomic_store(i64 16, i8* nonnull bitcast (%"struct.std::__1::atomic.40"* @gv_i128 to i8*), i8* nonnull %3, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z27atomic_load_relaxed_gv_u128o(i128 %0) {
; CHECK-LABEL: _Z27atomic_load_relaxed_gv_u128o:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s1, 248(, %s11)
; CHECK-NEXT:    st %s0, 240(, %s11)
; CHECK-NEXT:    lea %s0, __atomic_store@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __atomic_store@hi(, %s0)
; CHECK-NEXT:    lea %s0, gv_u128@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s1, gv_u128@hi(, %s0)
; CHECK-NEXT:    lea %s2, 240(, %s11)
; CHECK-NEXT:    or %s0, 16, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = alloca i128, align 16
  %3 = bitcast i128* %2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %3)
  store i128 %0, i128* %2, align 16, !tbaa !2
  call void @__atomic_store(i64 16, i8* nonnull bitcast (%"struct.std::__1::atomic.45"* @gv_u128 to i8*), i8* nonnull %3, i32 signext 0)
  call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind willreturn
declare void @__atomic_store(i64, i8*, i8*, i32)

!2 = !{!3, !3, i64 0}
!3 = !{!"__int128", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
