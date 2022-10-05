; RUN: llc < %s -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown | FileCheck %s
define i8 @atomic_min_i8() {
    top:
      %0 = alloca i8, align 2
      %1 = bitcast i8* %0 to i8*
      call void @llvm.lifetime.start.p0i8(i64 2, i8* %1)
      store i8 -1, i8* %0, align 2
      %2 = atomicrmw min i8* %0, i8 0 acq_rel
      %3 = load atomic i8, i8* %0 acquire, align 8
      call void @llvm.lifetime.end.p0i8(i64 2, i8* %1)
      ret i8 %3
; CHECK-LABEL: atomic_min_i8
; CHECK: lbarx [[DST:[0-9]+]],
; CHECK-NEXT: extsb [[EXT:[0-9]+]], [[DST]]
; CHECK-NEXT: cmpw [[EXT]], {{[0-9]+}}
; CHECK-NEXT: blt 0
}
define i16 @atomic_min_i16() {
    top:
      %0 = alloca i16, align 2
      %1 = bitcast i16* %0 to i8*
      call void @llvm.lifetime.start.p0i8(i64 2, i8* %1)
      store i16 -1, i16* %0, align 2
      %2 = atomicrmw min i16* %0, i16 0 acq_rel
      %3 = load atomic i16, i16* %0 acquire, align 8
      call void @llvm.lifetime.end.p0i8(i64 2, i8* %1)
      ret i16 %3
; CHECK-LABEL: atomic_min_i16
; CHECK: lharx [[DST:[0-9]+]],
; CHECK-NEXT: extsh [[EXT:[0-9]+]], [[DST]]
; CHECK-NEXT: cmpw [[EXT]], {{[0-9]+}}
; CHECK-NEXT: blt 0
}

define i8 @atomic_max_i8() {
    top:
      %0 = alloca i8, align 2
      %1 = bitcast i8* %0 to i8*
      call void @llvm.lifetime.start.p0i8(i64 2, i8* %1)
      store i8 -1, i8* %0, align 2
      %2 = atomicrmw max i8* %0, i8 0 acq_rel
      %3 = load atomic i8, i8* %0 acquire, align 8
      call void @llvm.lifetime.end.p0i8(i64 2, i8* %1)
      ret i8 %3
; CHECK-LABEL: atomic_max_i8
; CHECK: lbarx [[DST:[0-9]+]],
; CHECK-NEXT: extsb [[EXT:[0-9]+]], [[DST]]
; CHECK-NEXT: cmpw [[EXT]], {{[0-9]+}}
; CHECK-NEXT: bgt 0
}
define i16 @atomic_max_i16() {
    top:
      %0 = alloca i16, align 2
      %1 = bitcast i16* %0 to i8*
      call void @llvm.lifetime.start.p0i8(i64 2, i8* %1)
      store i16 -1, i16* %0, align 2
      %2 = atomicrmw max i16* %0, i16 0 acq_rel
      %3 = load atomic i16, i16* %0 acquire, align 8
      call void @llvm.lifetime.end.p0i8(i64 2, i8* %1)
      ret i16 %3
; CHECK-LABEL: atomic_max_i16
; CHECK: lharx [[DST:[0-9]+]],
; CHECK-NEXT: extsh [[EXT:[0-9]+]], [[DST]]
; CHECK-NEXT: cmpw [[EXT]], {{[0-9]+}}
; CHECK-NEXT: bgt 0
}

declare void @llvm.lifetime.start.p0i8(i64, i8*)
declare void @llvm.lifetime.end.p0i8(i64, i8*)
