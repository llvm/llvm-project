; RUN: llc -mtriple=arm64ec-pc-windows-msvc < %s | FileCheck %s

define void @f1(ptr %p, i64 %n) {
; CHECK-LABEL: "#f1":
; CHECK: bl "#memset"
  call void @llvm.memset.p0.i64(ptr %p, i8 0, i64 %n, i1 false)
  ret void
}

define void @f2(ptr %p1, ptr %p2, i64 %n) {
; CHECK-LABEL: "#f2":
; CHECK: bl "#memcpy"
  call void @llvm.memcpy.p0.i64(ptr %p1, ptr %p2, i64 %n, i1 false)
  ret void
}

define double @f3(double %x, double %y) {
; CHECK-LABEL: "#f3":
; CHECK: b "#fmod"
  %r = frem double %x, %y
  ret double %r
}

define i128 @f4(i128 %x, i128 %y) {
; CHECK-LABEL: "#f4":
; CHECK: bl "#__divti3"
  %r = sdiv i128 %x, %y
  ret i128 %r
}

define i8 @f5(i8 %expected, i8 %new, ptr %ptr) "target-features"="+outline-atomics" {
; CHECK-LABEL: "#f5":
; CHECK: bl "#__aarch64_cas1_relax"
    %pair = cmpxchg ptr %ptr, i8 %expected, i8 %new monotonic monotonic, align 1
   %r = extractvalue { i8, i1 } %pair, 0
    ret i8 %r
}

define float @f6(float %val, i32 %a) {
; CHECK-LABEL: "#f6":
; CHECK: bl "#ldexp"
  %call = tail call fast float @llvm.ldexp.f32(float %val, i32 %a)
  ret float %call
}

@dst = global [512 x i8] zeroinitializer, align 1
@src = global [512 x i8] zeroinitializer, align 1

define void @call__arm_sc_memcpy(i64 noundef %n) #0 {
; CHECK-LABEL: "#call__arm_sc_memcpy":
; CHECK: bl "#__arm_sc_memcpy"

  tail call void @llvm.memcpy.p0.p0.i64(ptr align 1 @dst, ptr nonnull align 1 @src, i64 %n, i1 false)
  ret void
}

define void @call__arm_sc_memmove(i64 noundef %n) #0 {
; CHECK-LABEL: "#call__arm_sc_memmove":
; CHECK: bl "#__arm_sc_memmove"
  tail call void @llvm.memmove.p0.p0.i64(ptr align 1 @dst, ptr nonnull align 1 @src, i64 %n, i1 false)
  ret void
}

define void @call__arm_sc_memset(i64 noundef %n) #0 {
; CHECK-LABEL: "#call__arm_sc_memset":
; CHECK: bl "#__arm_sc_memset"
  tail call void @llvm.memset.p0.i64(ptr align 1 @dst, i8 2, i64 %n, i1 false)
  ret void
}

attributes #0 = { nounwind "aarch64_pstate_sm_enabled" "target-features"="+sme2" }
