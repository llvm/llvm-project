; RUN: llc < %s -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown | FileCheck %s
define i8 @atomic_min_i8() {
    top:
      %0 = alloca i8, align 2
      call void @llvm.lifetime.start.p0(i64 2, ptr %0)
      store i8 -1, ptr %0, align 2
      %1 = atomicrmw min ptr %0, i8 0 acq_rel
      %2 = load atomic i8, ptr %0 acquire, align 8
      call void @llvm.lifetime.end.p0(i64 2, ptr %0)
      ret i8 %2
; CHECK-LABEL: atomic_min_i8
; CHECK: lbarx [[DST:[0-9]+]],
; CHECK-NEXT: extsb [[EXT:[0-9]+]], [[DST]]
; CHECK-NEXT: cmpw [[EXT]], {{[0-9]+}}
; CHECK-NEXT: blt 0
}
define i16 @atomic_min_i16() {
    top:
      %0 = alloca i16, align 2
      call void @llvm.lifetime.start.p0(i64 2, ptr %0)
      store i16 -1, ptr %0, align 2
      %1 = atomicrmw min ptr %0, i16 0 acq_rel
      %2 = load atomic i16, ptr %0 acquire, align 8
      call void @llvm.lifetime.end.p0(i64 2, ptr %0)
      ret i16 %2
; CHECK-LABEL: atomic_min_i16
; CHECK: lharx [[DST:[0-9]+]],
; CHECK-NEXT: extsh [[EXT:[0-9]+]], [[DST]]
; CHECK-NEXT: cmpw [[EXT]], {{[0-9]+}}
; CHECK-NEXT: blt 0
}

define i8 @atomic_max_i8() {
    top:
      %0 = alloca i8, align 2
      call void @llvm.lifetime.start.p0(i64 2, ptr %0)
      store i8 -1, ptr %0, align 2
      %1 = atomicrmw max ptr %0, i8 0 acq_rel
      %2 = load atomic i8, ptr %0 acquire, align 8
      call void @llvm.lifetime.end.p0(i64 2, ptr %0)
      ret i8 %2
; CHECK-LABEL: atomic_max_i8
; CHECK: lbarx [[DST:[0-9]+]],
; CHECK-NEXT: extsb [[EXT:[0-9]+]], [[DST]]
; CHECK-NEXT: cmpw [[EXT]], {{[0-9]+}}
; CHECK-NEXT: bgt 0
}
define i16 @atomic_max_i16() {
    top:
      %0 = alloca i16, align 2
      call void @llvm.lifetime.start.p0(i64 2, ptr %0)
      store i16 -1, ptr %0, align 2
      %1 = atomicrmw max ptr %0, i16 0 acq_rel
      %2 = load atomic i16, ptr %0 acquire, align 8
      call void @llvm.lifetime.end.p0(i64 2, ptr %0)
      ret i16 %2
; CHECK-LABEL: atomic_max_i16
; CHECK: lharx [[DST:[0-9]+]],
; CHECK-NEXT: extsh [[EXT:[0-9]+]], [[DST]]
; CHECK-NEXT: cmpw [[EXT]], {{[0-9]+}}
; CHECK-NEXT: bgt 0
}

declare void @llvm.lifetime.start.p0(i64, ptr)
declare void @llvm.lifetime.end.p0(i64, ptr)
