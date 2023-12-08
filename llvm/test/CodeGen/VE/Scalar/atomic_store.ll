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
define void @_Z23atomic_store_relaxed_i1RNSt3__16atomicIbEEb(ptr nocapture nonnull align 1 dereferenceable(1) %0, i1 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_relaxed_i1RNSt3__16atomicIbEEb:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i1 %1 to i8
  store atomic i8 %3, ptr %0 monotonic, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_relaxed_i8RNSt3__16atomicIcEEc(ptr nocapture nonnull align 1 dereferenceable(1) %0, i8 signext %1) {
; CHECK-LABEL: _Z23atomic_store_relaxed_i8RNSt3__16atomicIcEEc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i8 %1, ptr %0 monotonic, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_relaxed_u8RNSt3__16atomicIhEEh(ptr nocapture nonnull align 1 dereferenceable(1) %0, i8 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_relaxed_u8RNSt3__16atomicIhEEh:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i8 %1, ptr %0 monotonic, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_i16RNSt3__16atomicIsEEs(ptr nocapture nonnull align 2 dereferenceable(2) %0, i16 signext %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_i16RNSt3__16atomicIsEEs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i16 %1, ptr %0 monotonic, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_u16RNSt3__16atomicItEEt(ptr nocapture nonnull align 2 dereferenceable(2) %0, i16 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_u16RNSt3__16atomicItEEt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i16 %1, ptr %0 monotonic, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_i32RNSt3__16atomicIiEEi(ptr nocapture nonnull align 4 dereferenceable(4) %0, i32 signext %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_i32RNSt3__16atomicIiEEi:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i32 %1, ptr %0 monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_u32RNSt3__16atomicIjEEj(ptr nocapture nonnull align 4 dereferenceable(4) %0, i32 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_u32RNSt3__16atomicIjEEj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i32 %1, ptr %0 monotonic, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_i64RNSt3__16atomicIlEEl(ptr nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_i64RNSt3__16atomicIlEEl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i64 %1, ptr %0 monotonic, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_relaxed_u64RNSt3__16atomicImEEm(ptr nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_relaxed_u64RNSt3__16atomicImEEm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i64 %1, ptr %0 monotonic, align 8
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_relaxed_i128RNSt3__16atomicInEEn(ptr nonnull align 16 dereferenceable(16) %0, i128 %1) {
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
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  store i128 %1, ptr %3, align 16, !tbaa !2
  call void @__atomic_store(i64 16, ptr nonnull %0, ptr nonnull %3, i32 signext 0)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_relaxed_u128RNSt3__16atomicIoEEo(ptr nonnull align 16 dereferenceable(16) %0, i128 %1) {
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
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  store i128 %1, ptr %3, align 16, !tbaa !2
  call void @__atomic_store(i64 16, ptr nonnull %0, ptr nonnull %3, i32 signext 0)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_release_i1RNSt3__16atomicIbEEb(ptr nocapture nonnull align 1 dereferenceable(1) %0, i1 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_release_i1RNSt3__16atomicIbEEb:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i1 %1 to i8
  store atomic i8 %3, ptr %0 release, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_release_i8RNSt3__16atomicIcEEc(ptr nocapture nonnull align 1 dereferenceable(1) %0, i8 signext %1) {
; CHECK-LABEL: _Z23atomic_store_release_i8RNSt3__16atomicIcEEc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i8 %1, ptr %0 release, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_release_u8RNSt3__16atomicIhEEh(ptr nocapture nonnull align 1 dereferenceable(1) %0, i8 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_release_u8RNSt3__16atomicIhEEh:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i8 %1, ptr %0 release, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_i16RNSt3__16atomicIsEEs(ptr nocapture nonnull align 2 dereferenceable(2) %0, i16 signext %1) {
; CHECK-LABEL: _Z24atomic_store_release_i16RNSt3__16atomicIsEEs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i16 %1, ptr %0 release, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_u16RNSt3__16atomicItEEt(ptr nocapture nonnull align 2 dereferenceable(2) %0, i16 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_release_u16RNSt3__16atomicItEEt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i16 %1, ptr %0 release, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_i32RNSt3__16atomicIiEEi(ptr nocapture nonnull align 4 dereferenceable(4) %0, i32 signext %1) {
; CHECK-LABEL: _Z24atomic_store_release_i32RNSt3__16atomicIiEEi:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i32 %1, ptr %0 release, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_u32RNSt3__16atomicIjEEj(ptr nocapture nonnull align 4 dereferenceable(4) %0, i32 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_release_u32RNSt3__16atomicIjEEj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i32 %1, ptr %0 release, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_i64RNSt3__16atomicIlEEl(ptr nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_release_i64RNSt3__16atomicIlEEl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i64 %1, ptr %0 release, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_release_u64RNSt3__16atomicImEEm(ptr nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_release_u64RNSt3__16atomicImEEm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i64 %1, ptr %0 release, align 8
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_release_i128RNSt3__16atomicInEEn(ptr nonnull align 16 dereferenceable(16) %0, i128 %1) {
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
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  store i128 %1, ptr %3, align 16, !tbaa !2
  call void @__atomic_store(i64 16, ptr nonnull %0, ptr nonnull %3, i32 signext 3)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_release_u128RNSt3__16atomicIoEEo(ptr nonnull align 16 dereferenceable(16) %0, i128 %1) {
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
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  store i128 %1, ptr %3, align 16, !tbaa !2
  call void @__atomic_store(i64 16, ptr nonnull %0, ptr nonnull %3, i32 signext 3)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_seq_cst_i1RNSt3__16atomicIbEEb(ptr nocapture nonnull align 1 dereferenceable(1) %0, i1 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_seq_cst_i1RNSt3__16atomicIbEEb:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  %3 = zext i1 %1 to i8
  store atomic i8 %3, ptr %0 seq_cst, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_seq_cst_i8RNSt3__16atomicIcEEc(ptr nocapture nonnull align 1 dereferenceable(1) %0, i8 signext %1) {
; CHECK-LABEL: _Z23atomic_store_seq_cst_i8RNSt3__16atomicIcEEc:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i8 %1, ptr %0 seq_cst, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z23atomic_store_seq_cst_u8RNSt3__16atomicIhEEh(ptr nocapture nonnull align 1 dereferenceable(1) %0, i8 zeroext %1) {
; CHECK-LABEL: _Z23atomic_store_seq_cst_u8RNSt3__16atomicIhEEh:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st1b %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i8 %1, ptr %0 seq_cst, align 1
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_i16RNSt3__16atomicIsEEs(ptr nocapture nonnull align 2 dereferenceable(2) %0, i16 signext %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_i16RNSt3__16atomicIsEEs:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i16 %1, ptr %0 seq_cst, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_u16RNSt3__16atomicItEEt(ptr nocapture nonnull align 2 dereferenceable(2) %0, i16 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_u16RNSt3__16atomicItEEt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st2b %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i16 %1, ptr %0 seq_cst, align 2
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_i32RNSt3__16atomicIiEEi(ptr nocapture nonnull align 4 dereferenceable(4) %0, i32 signext %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_i32RNSt3__16atomicIiEEi:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i32 %1, ptr %0 seq_cst, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_u32RNSt3__16atomicIjEEj(ptr nocapture nonnull align 4 dereferenceable(4) %0, i32 zeroext %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_u32RNSt3__16atomicIjEEj:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    stl %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i32 %1, ptr %0 seq_cst, align 4
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_i64RNSt3__16atomicIlEEl(ptr nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_i64RNSt3__16atomicIlEEl:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i64 %1, ptr %0 seq_cst, align 8
  ret void
}

; Function Attrs: nofree norecurse nounwind mustprogress
define void @_Z24atomic_store_seq_cst_u64RNSt3__16atomicImEEm(ptr nocapture nonnull align 8 dereferenceable(8) %0, i64 %1) {
; CHECK-LABEL: _Z24atomic_store_seq_cst_u64RNSt3__16atomicImEEm:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  store atomic i64 %1, ptr %0 seq_cst, align 8
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_seq_cst_i128RNSt3__16atomicInEEn(ptr nonnull align 16 dereferenceable(16) %0, i128 %1) {
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
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  store i128 %1, ptr %3, align 16, !tbaa !2
  call void @__atomic_store(i64 16, ptr nonnull %0, ptr nonnull %3, i32 signext 5)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
  ret void
}

; Function Attrs: nofree nounwind mustprogress
define void @_Z25atomic_store_seq_cst_u128RNSt3__16atomicIoEEo(ptr nonnull align 16 dereferenceable(16) %0, i128 %1) {
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
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  store i128 %1, ptr %3, align 16, !tbaa !2
  call void @__atomic_store(i64 16, ptr nonnull %0, ptr nonnull %3, i32 signext 5)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
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
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %2)
  %3 = zext i1 %0 to i8
  store atomic volatile i8 %3, ptr %2 monotonic, align 1
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %2)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

; Function Attrs: nofree nounwind mustprogress
define void @_Z26atomic_load_relaxed_stk_i8c(i8 signext %0) {
; CHECK-LABEL: _Z26atomic_load_relaxed_stk_i8c:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st1b %s0, 15(, %s11)
; CHECK-NEXT:    adds.l %s11, 16, %s11
; CHECK-NEXT:    b.l.t (, %s10)
  %2 = alloca i8, align 1
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %2)
  store atomic volatile i8 %0, ptr %2 monotonic, align 1
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %2)
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
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %2)
  store atomic volatile i8 %0, ptr %2 monotonic, align 1
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %2)
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
  call void @llvm.lifetime.start.p0(i64 2, ptr nonnull %2)
  store atomic volatile i16 %0, ptr %2 monotonic, align 2
  call void @llvm.lifetime.end.p0(i64 2, ptr nonnull %2)
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
  call void @llvm.lifetime.start.p0(i64 2, ptr nonnull %2)
  store atomic volatile i16 %0, ptr %2 monotonic, align 2
  call void @llvm.lifetime.end.p0(i64 2, ptr nonnull %2)
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
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %2)
  store atomic volatile i32 %0, ptr %2 monotonic, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %2)
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
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %2)
  store atomic volatile i32 %0, ptr %2 monotonic, align 4
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %2)
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
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %2)
  store atomic volatile i64 %0, ptr %2 monotonic, align 8
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %2)
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
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %2)
  store atomic volatile i64 %0, ptr %2 monotonic, align 8
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %2)
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
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %2)
  store i128 %0, ptr %2, align 16, !tbaa !2
  call void @__atomic_store(i64 16, ptr nonnull %3, ptr nonnull %2, i32 signext 0)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %2)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
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
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %3)
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %2)
  store i128 %0, ptr %2, align 16, !tbaa !2
  call void @__atomic_store(i64 16, ptr nonnull %3, ptr nonnull %2, i32 signext 0)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %2)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %3)
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
  store atomic i8 %2, ptr @gv_i1 monotonic, align 4
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
  store atomic i8 %0, ptr @gv_i8 monotonic, align 4
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
  store atomic i8 %0, ptr @gv_u8 monotonic, align 4
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
  store atomic i16 %0, ptr @gv_i16 monotonic, align 4
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
  store atomic i16 %0, ptr @gv_u16 monotonic, align 4
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
  store atomic i32 %0, ptr @gv_i32 monotonic, align 4
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
  store atomic i32 %0, ptr @gv_u32 monotonic, align 4
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
  store atomic i64 %0, ptr @gv_i64 monotonic, align 8
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
  store atomic i64 %0, ptr @gv_u64 monotonic, align 8
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
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %2)
  store i128 %0, ptr %2, align 16, !tbaa !2
  call void @__atomic_store(i64 16, ptr nonnull @gv_i128, ptr nonnull %2, i32 signext 0)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %2)
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
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %2)
  store i128 %0, ptr %2, align 16, !tbaa !2
  call void @__atomic_store(i64 16, ptr nonnull @gv_u128, ptr nonnull %2, i32 signext 0)
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %2)
  ret void
}

; Function Attrs: nofree nounwind willreturn
declare void @__atomic_store(i64, ptr, ptr, i32)

!2 = !{!3, !3, i64 0}
!3 = !{!"__int128", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
