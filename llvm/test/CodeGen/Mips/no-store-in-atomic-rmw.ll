; RUN: llc -O0 -march=mips64 -mcpu=mips64r2 < %s | FileCheck %s

; Check that no stores occur between ll and sc when the fast register allocator
; is used. Atomic read-modify-write sequences on certain MIPS implementations
; will fail if a store occurs between a ll and sc.

define i32 @main() {
; CHECK-LABEL: main:
entry:
  %retval = alloca i32, align 4
  %I = alloca i32, align 4
  %k = alloca i32, align 4
  %i = alloca i32*, align 8
  %ret = alloca i32, align 4
  %flag_k = alloca i8, align 1
  %.atomictmp = alloca i32, align 4
  %atomic-temp = alloca i32, align 4
  %.atomictmp1 = alloca i32, align 4
  %atomic-temp2 = alloca i32, align 4
  %.atomictmp3 = alloca i32, align 4
  %atomic-temp4 = alloca i32, align 4
  %.atomictmp5 = alloca i32, align 4
  %atomic-temp6 = alloca i32, align 4
  %.atomictmp7 = alloca i32, align 4
  %atomic-temp8 = alloca i32, align 4
  %.atomictmp9 = alloca i32, align 4
  %atomic-temp10 = alloca i32, align 4
  %.atomictmp11 = alloca i32, align 4
  %atomic-temp12 = alloca i32, align 4
  %.atomictmp13 = alloca i32, align 4
  %cmpxchg.bool = alloca i8, align 1
  %cmpxchg.bool14 = alloca i8, align 1
  store i32 0, i32* %retval, align 4
  store i32 0, i32* %I, align 4
  store i32 5, i32* %k, align 4
  store i32* %I, i32** %i, align 8
  store i32 0, i32* %ret, align 4
  store i8 0, i8* %flag_k, align 1
  %0 = load i32*, i32** %i, align 8
  %1 = load i32, i32* %k, align 4
  %2 = atomicrmw xchg i32* %0, i32 %1 monotonic
; CHECK-LABEL: .LBB0_1:
; CHECK: ll
; CHECK-NOT: sd
; CHECK-NOT: sw
; CHECK: sc
  store i32 %2, i32* %ret, align 4
  %3 = load i32*, i32** %i, align 8
  store i32 3, i32* %.atomictmp, align 4
  %4 = load i32, i32* %.atomictmp, align 4
  %5 = atomicrmw add i32* %3, i32 %4 monotonic
; CHECK-LABEL: .LBB0_3:
; CHECK: ll
; CHECK-NOT: sd
; CHECK-NOT: sw
; CHECK: addu
; CHECK: sc
  store i32 %5, i32* %atomic-temp, align 4
  %6 = load i32, i32* %atomic-temp, align 4
  %7 = load i32*, i32** %i, align 8
  store i32 3, i32* %.atomictmp1, align 4
  %8 = load i32, i32* %.atomictmp1, align 4
  %9 = atomicrmw sub i32* %7, i32 %8 monotonic
; CHECK-LABEL: .LBB0_5:
; CHECK: ll
; CHECK-NOT: sd
; CHECK-NOT: sw
; CHECK: subu
; CHECK: sc
  store i32 %9, i32* %atomic-temp2, align 4
  %10 = load i32, i32* %atomic-temp2, align 4
  %11 = load i32*, i32** %i, align 8
  store i32 3, i32* %.atomictmp3, align 4
  %12 = load i32, i32* %.atomictmp3, align 4
  %13 = atomicrmw and i32* %11, i32 %12 monotonic
; CHECK-LABEL: .LBB0_7:
; CHECK: ll
; CHECK-NOT: sd
; CHECK-NOT: sw
; CHECK: and
; CHECK: sc
  store i32 %13, i32* %atomic-temp4, align 4
  %14 = load i32, i32* %atomic-temp4, align 4
  %15 = load i32*, i32** %i, align 8
  store i32 3, i32* %.atomictmp5, align 4
  %16 = load i32, i32* %.atomictmp5, align 4
  %17 = atomicrmw or i32* %15, i32 %16 monotonic
; CHECK-LABEL: .LBB0_9:
; CHECK: ll
; CHECK-NOT: sd
; CHECK-NOT: sw
; CHECK: or
; CHECK: sc
  %18 = load i32*, i32** %i, align 8
  store i32 5, i32* %.atomictmp13, align 4
  %19 = load i32, i32* %I, align 4
  %20 = load i32, i32* %.atomictmp13, align 4
  %21 = cmpxchg weak i32* %18, i32 %19, i32 %20 monotonic monotonic
; CHECK-LABEL: .LBB0_11:
; CHECK: ll
; CHECK-NOT: sd
; CHECK-NOT: sw
; CHECK: sc
  %22 = extractvalue { i32, i1 } %21, 0
  %23 = extractvalue { i32, i1 } %21, 1
  br i1 %23, label %cmpxchg.continue, label %cmpxchg.store_expected

cmpxchg.store_expected:                           ; preds = %entry
  store i32 %22, i32* %I, align 4
  br label %cmpxchg.continue

cmpxchg.continue:                                 ; preds = %cmpxchg.store_expected, %entry
  %frombool = zext i1 %23 to i8
  store i8 %frombool, i8* %cmpxchg.bool, align 1
  %24 = load i8, i8* %cmpxchg.bool, align 1
  %tobool = trunc i8 %24 to i1
  %25 = load i32*, i32** %i, align 8
  %26 = load i32, i32* %I, align 4
  %27 = load i32, i32* %ret, align 4
  %28 = cmpxchg i32* %25, i32 %26, i32 %27 monotonic monotonic
; CHECK-LABEL: .LBB0_17:
; CHECK: ll
; CHECK-NOT: sd
; CHECK-NOT: sw
; CHECK: sc
  %29 = extractvalue { i32, i1 } %28, 0
  %30 = extractvalue { i32, i1 } %28, 1
  br i1 %30, label %cmpxchg.continue16, label %cmpxchg.store_expected15

cmpxchg.store_expected15:                         ; preds = %cmpxchg.continue
  store i32 %29, i32* %I, align 4
  br label %cmpxchg.continue16

cmpxchg.continue16:                               ; preds = %cmpxchg.store_expected15, %cmpxchg.continue
  %frombool17 = zext i1 %30 to i8
  store i8 %frombool17, i8* %cmpxchg.bool14, align 1
  %31 = load i8, i8* %cmpxchg.bool14, align 1
  %tobool18 = trunc i8 %31 to i1
  %32 = atomicrmw xchg i8* %flag_k, i8 1 monotonic
; CHECK-LABEL: .LBB0_23:
; CHECK: ll
; CHECK-NOT: sd
; CHECK-NOT: sw
; CHECK: sc
  %tobool19 = icmp ne i8 %32, 0
  %33 = atomicrmw xchg i8* %flag_k, i8 1 monotonic
; CHECK-LABEL: .LBB0_26:
; CHECK: ll
; CHECK-NOT: sd
; CHECK-NOT: sw
; CHECK: sc
  %tobool20 = icmp ne i8 %33, 0
  store atomic i8 0, i8* %flag_k monotonic, align 1
  %34 = load i32, i32* %retval, align 4
  ret i32 %34
}
