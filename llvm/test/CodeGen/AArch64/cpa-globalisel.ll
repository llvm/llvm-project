; RUN: llc -mtriple=aarch64 -verify-machineinstrs --mattr=+cpa,cpa-codegen -O0 -global-isel=1 -global-isel-abort=1 %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK-CPA-O0
; RUN: llc -mtriple=aarch64 -verify-machineinstrs --mattr=+cpa,cpa-codegen -O3 -global-isel=1 -global-isel-abort=1 %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK-CPA-O3
; RUN: llc -mtriple=aarch64 -verify-machineinstrs --mattr=-cpa -O0 -global-isel=1 -global-isel-abort=1 %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK-NOCPA-O0
; RUN: llc -mtriple=aarch64 -verify-machineinstrs --mattr=-cpa -O3 -global-isel=1 -global-isel-abort=1 %s -o - 2>&1 | FileCheck %s --check-prefixes=CHECK-NOCPA-O3

%struct.my_type = type { i64, i64 }
%struct.my_type2 = type { i64, i64, i64, i64, i64, i64 }

@array = external dso_local global [10 x %struct.my_type], align 8
@array2 = external dso_local global [10 x %struct.my_type2], align 8

define void @addpt1(i64 %index, i64 %arg) {
; CHECK-CPA-O0-LABEL:    addpt1:
; CHECK-CPA-O0:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-CPA-O0:          str	x{{[0-9]+}}, [[[REG1]], #8]
;
; CHECK-CPA-O3-LABEL:    addpt1:
; CHECK-CPA-O3:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-CPA-O3:          str	x{{[0-9]+}}, [[[REG1]], #8]
;
; CHECK-NOCPA-O0-LABEL:  addpt1:
; CHECK-NOCPA-O0:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-NOCPA-O0:        str	x{{[0-9]+}}, [[[REG1]], #8]
;
; CHECK-NOCPA-O3-LABEL:  addpt1:
; CHECK-NOCPA-O3:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-NOCPA-O3:        str	x{{[0-9]+}}, [[[REG1]], #8]
entry:
  %e2 = getelementptr inbounds %struct.my_type, ptr @array, i64 %index, i32 1
  store i64 %arg, ptr %e2, align 8
  ret void
}

define void @maddpt1(i32 %pos, ptr %val) {
; CHECK-CPA-O0-LABEL:    maddpt1:
; CHECK-CPA-O0:          maddpt	x0, x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O0:          b	memcpy
;
; CHECK-CPA-O3-LABEL:    maddpt1:
; CHECK-CPA-O3:          maddpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]]]
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #16]
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #32]
;
; CHECK-NOCPA-O0-LABEL:  maddpt1:
; CHECK-NOCPA-O0:        smaddl	x0, w{{[0-9]+}}, w{{[0-9]+}}, x{{[0-9]+}}
; CHECK-NOCPA-O0:        b	memcpy
;
; CHECK-NOCPA-O3-LABEL:  maddpt1:
; CHECK-NOCPA-O3:        smaddl	[[REG1:x[0-9]+]], w{{[0-9]+}}, w{{[0-9]+}}, x{{[0-9]+}}
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]]]
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #16]
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #32]
entry:
  %idxprom = sext i32 %pos to i64
  %arrayidx = getelementptr inbounds [10 x %struct.my_type2], ptr @array2, i64 0, i64 %idxprom
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 dereferenceable(48) %arrayidx, ptr align 8 dereferenceable(48) %val, i64 48, i1 false)
  ret void
}

define void @msubpt1(i32 %index, i32 %elem) {
; CHECK-CPA-O0-LABEL:    msubpt1:
; CHECK-CPA-O0:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O0:          msubpt	x0, x{{[0-9]+}}, x{{[0-9]+}}, [[REG1]]
; CHECK-CPA-O0:          addpt	x1, x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O0:          b	memcpy
;
; CHECK-CPA-O3-LABEL:    msubpt1:
; CHECK-CPA-O3:          msubpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #192]
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #208]
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #224]
;
; CHECK-NOCPA-O0-LABEL:  msubpt1:
; CHECK-NOCPA-O0:        mneg	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-NOCPA-O0:        add	x0, x{{[0-9]+}}, [[REG1]]
; CHECK-NOCPA-O0:        b	memcpy
;
; CHECK-NOCPA-O3-LABEL:  msubpt1:
; CHECK-NOCPA-O3:        mneg	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-NOCPA-O3:        add	[[REG2:x[0-9]+]], x{{[0-9]+}}, [[REG1]]
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #192]
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #208]
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #224]
entry:
  %idx.ext = sext i32 %index to i64
  %idx.neg = sub nsw i64 0, %idx.ext
  %add.ptr = getelementptr inbounds %struct.my_type2, ptr getelementptr inbounds ([10 x %struct.my_type2], ptr @array2, i64 0, i64 6), i64 %idx.neg
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 8 dereferenceable(48) %add.ptr, ptr align 8 dereferenceable(48) getelementptr inbounds ([10 x %struct.my_type2], ptr @array2, i64 0, i64 2), i64 48, i1 false), !tbaa.struct !6
  ret void
}

define void @subpt1(i32 %index, i32 %elem) {
; CHECK-CPA-O0-LABEL:    subpt1:
; CHECK-CPA-O0:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O0:          str	q{{[0-9]+}}, [[[REG1]], x{{[0-9]+}}, lsl #4]
;
; CHECK-CPA-O3-LABEL:    subpt1:
; CHECK-CPA-O3:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #64]
;
; CHECK-NOCPA-O0-LABEL:  subpt1:
; CHECK-NOCPA-O0:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, #96
; CHECK-NOCPA-O0:        str	q{{[0-9]+}}, [[[REG1]], x{{[0-9]+}}, lsl #4]
;
; CHECK-NOCPA-O3-LABEL:  subpt1:
; CHECK-NOCPA-O3:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #64]
entry:
  %conv = sext i32 %index to i64
  %mul.neg = mul nsw i64 %conv, -16
  %add.ptr = getelementptr inbounds %struct.my_type, ptr getelementptr inbounds ([10 x %struct.my_type], ptr @array, i64 0, i64 6), i64 %mul.neg
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %add.ptr, ptr noundef nonnull align 8 dereferenceable(16) getelementptr inbounds ([10 x %struct.my_type], ptr @array, i64 0, i64 2), i64 16, i1 false), !tbaa.struct !6
  ret void
}

define void @subpt2(i32 %index, i32 %elem) {
; CHECK-CPA-O0-LABEL:    subpt2:
; CHECK-CPA-O0:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}
; CHECK-CPA-O0:          str	q{{[0-9]+}}, [[[REG1]], x{{[0-9]+}}, lsl #4]
;
; CHECK-CPA-O3-LABEL:    subpt2:
; CHECK-CPA-O3:          addpt	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-CPA-O3:          str	q{{[0-9]+}}, [[[REG1]], #64]
;
; CHECK-NOCPA-O0-LABEL:  subpt2:
; CHECK-NOCPA-O0:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, #96
; CHECK-NOCPA-O0:        str	q{{[0-9]+}}, [[[REG1]], x{{[0-9]+}}, lsl #4]
;
; CHECK-NOCPA-O3-LABEL:  subpt2:
; CHECK-NOCPA-O3:        add	[[REG1:x[0-9]+]], x{{[0-9]+}}, x{{[0-9]+}}, lsl #4
; CHECK-NOCPA-O3:        str	q{{[0-9]+}}, [[[REG1]], #64]
entry:
  %idx.ext = sext i32 %index to i64
  %idx.neg = sub nsw i64 0, %idx.ext
  %add.ptr = getelementptr inbounds %struct.my_type, ptr getelementptr inbounds ([10 x %struct.my_type], ptr @array, i64 0, i64 6), i64 %idx.neg
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %add.ptr, ptr noundef nonnull align 8 dereferenceable(16) getelementptr inbounds ([10 x %struct.my_type], ptr @array, i64 0, i64 2), i64 16, i1 false), !tbaa.struct !11
  ret void
}

define ptr @subpt3(ptr %ptr, i32 %index) {
; CHECK-CPA-O0-LABEL:    subpt3:
; CHECK-CPA-O0:          mov	[[REG1:x[0-9]+]], #-8
; CHECK-CPA-O0:          addpt	x0, x{{[0-9]+}}, [[REG1]]
; CHECK-CPA-O0:          ret
;
; CHECK-CPA-O3-LABEL:    subpt3:
; CHECK-CPA-O3:          mov	[[REG1:x[0-9]+]], #-8
; CHECK-CPA-O3:          addpt	x0, x{{[0-9]+}}, [[REG1]]
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  subpt3:
; CHECK-NOCPA-O0:        subs	x0, x{{[0-9]+}}, #8
; CHECK-NOCPA-O0:        ret
;
; CHECK-NOCPA-O3-LABEL:  subpt3:
; CHECK-NOCPA-O3:        sub	x0, x{{[0-9]+}}, #8
; CHECK-NOCPA-O3:        ret
entry:
  %incdec.ptr.i.i.i = getelementptr inbounds i64, ptr %ptr, i64 -1
  ret ptr %incdec.ptr.i.i.i
}

define i64 @subi64(i64 %ptr, i32 %index) {
; CHECK-CPA-O0-LABEL:    subi64:
; CHECK-CPA-O0:          subs x0, x0, #1
; CHECK-CPA-O0:          ret
;
; CHECK-CPA-O3-LABEL:    subi64:
; CHECK-CPA-O3:          sub x0, x0, #1
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  subi64:
; CHECK-NOCPA-O0:        subs x0, x0, #1
; CHECK-NOCPA-O0:        ret
;
; CHECK-NOCPA-O3-LABEL:  subi64:
; CHECK-NOCPA-O3:        sub x0, x0, #1
; CHECK-NOCPA-O3:        ret
entry:
  %incdec.ptr.i.i.i = add i64 %ptr, -1
  ret i64 %incdec.ptr.i.i.i
}

define i32 @subi32(i32 %ptr, i32 %index) {
; CHECK-CPA-O0-LABEL:    subi32:
; CHECK-CPA-O0:          subs w0, w0, #1
; CHECK-CPA-O0:          ret
;
; CHECK-CPA-O3-LABEL:    subi32:
; CHECK-CPA-O3:          sub w0, w0, #1
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  subi32:
; CHECK-NOCPA-O0:        subs w0, w0, #1
; CHECK-NOCPA-O0:        ret
;
; CHECK-NOCPA-O3-LABEL:  subi32:
; CHECK-NOCPA-O3:        sub w0, w0, #1
; CHECK-NOCPA-O3:        ret
entry:
  %incdec.ptr.i.i.i = add i32 %ptr, -1
  ret i32 %incdec.ptr.i.i.i
}

define i16 @subi16(i16 %ptr, i32 %index) {
; CHECK-CPA-O0-LABEL:    subi16:
; CHECK-CPA-O0:          subs w0, w0, #1
; CHECK-CPA-O0:          ret
;
; CHECK-CPA-O3-LABEL:    subi16:
; CHECK-CPA-O3:          sub w0, w0, #1
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  subi16:
; CHECK-NOCPA-O0:        subs w0, w0, #1
; CHECK-NOCPA-O0:        ret
;
; CHECK-NOCPA-O3-LABEL:  subi16:
; CHECK-NOCPA-O3:        sub w0, w0, #1
; CHECK-NOCPA-O3:        ret
entry:
  %incdec.ptr.i.i.i = add i16 %ptr, -1
  ret i16 %incdec.ptr.i.i.i
}

define i64 @addi64(i64 %ptr, i32 %index) {
; CHECK-CPA-O0-LABEL:    addi64:
; CHECK-CPA-O0:          add x0, x0, #1
; CHECK-CPA-O0:          ret
;
; CHECK-CPA-O3-LABEL:    addi64:
; CHECK-CPA-O3:          add x0, x0, #1
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  addi64:
; CHECK-NOCPA-O0:        add x0, x0, #1
; CHECK-NOCPA-O0:        ret
;
; CHECK-NOCPA-O3-LABEL:  addi64:
; CHECK-NOCPA-O3:        add x0, x0, #1
; CHECK-NOCPA-O3:        ret
entry:
  %incdec.ptr.i.i.i = add i64 %ptr, 1
  ret i64 %incdec.ptr.i.i.i
}

define i32 @addi32(i32 %ptr, i32 %index) {
; CHECK-CPA-O0-LABEL:    addi32:
; CHECK-CPA-O0:          add w0, w0, #1
; CHECK-CPA-O0:          ret
;
; CHECK-CPA-O3-LABEL:    addi32:
; CHECK-CPA-O3:          add w0, w0, #1
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  addi32:
; CHECK-NOCPA-O0:        add w0, w0, #1
; CHECK-NOCPA-O0:        ret
;
; CHECK-NOCPA-O3-LABEL:  addi32:
; CHECK-NOCPA-O3:        add w0, w0, #1
; CHECK-NOCPA-O3:        ret
entry:
  %incdec.ptr.i.i.i = add i32 %ptr, 1
  ret i32 %incdec.ptr.i.i.i
}

define i16 @addi16(i16 %ptr, i32 %index) {
; CHECK-CPA-O0-LABEL:    addi16:
; CHECK-CPA-O0:          add w0, w0, #1
; CHECK-CPA-O0:          ret
;
; CHECK-CPA-O3-LABEL:    addi16:
; CHECK-CPA-O3:          add w0, w0, #1
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  addi16:
; CHECK-NOCPA-O0:        add w0, w0, #1
; CHECK-NOCPA-O0:        ret
;
; CHECK-NOCPA-O3-LABEL:  addi16:
; CHECK-NOCPA-O3:        add w0, w0, #1
; CHECK-NOCPA-O3:        ret
entry:
  %incdec.ptr.i.i.i = add i16 %ptr, 1
  ret i16 %incdec.ptr.i.i.i
}

define i64 @arith1(i64 noundef %0, i64 noundef %1, i64 noundef %2) {
; CHECK-CPA-O0-LABEL:    arith1:
; CHECK-CPA-O0:          mul	x9, x9, x10
; CHECK-CPA-O0:          add	x0, x8, x9
; CHECK-CPA-O0:          add	sp, sp, #32
; CHECK-CPA-O0:          ret
;
; CHECK-CPA-O3-LABEL:    arith1:
; CHECK-CPA-O3:          madd	x0, x1, x2, x0
; CHECK-CPA-O3:          add	sp, sp, #32
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  arith1:
; CHECK-NOCPA-O0:        mul	x9, x9, x10
; CHECK-NOCPA-O0:        add	x0, x8, x9
; CHECK-NOCPA-O0:        add	sp, sp, #32
; CHECK-NOCPA-O0:        ret
;
; CHECK-NOCPA-O3-LABEL:  arith1:
; CHECK-NOCPA-O3:        madd	x0, x1, x2, x0
; CHECK-NOCPA-O3:        add	sp, sp, #32
; CHECK-NOCPA-O3:        ret
entry:
  %4 = alloca i64, align 8
  %5 = alloca i64, align 8
  %6 = alloca i64, align 8
  store i64 %0, ptr %4, align 8
  store i64 %1, ptr %5, align 8
  store i64 %2, ptr %6, align 8
  %7 = load i64, ptr %4, align 8
  %8 = load i64, ptr %5, align 8
  %9 = load i64, ptr %6, align 8
  %10 = mul nsw i64 %8, %9
  %11 = add nsw i64 %7, %10
  ret i64 %11
}

define i64 @arith2(ptr noundef %0, i64 noundef %1, i64 noundef %2, i32 noundef %3) {
; CHECK-CPA-O0-LABEL:    arith2:
; CHECK-CPA-O0:          maddpt	x8, x8, x9, x10
; CHECK-CPA-O0:          ldr	x8, [x8, #24]
; CHECK-CPA-O0:          ldr	x10, [sp, #16]
; CHECK-CPA-O0:          ldr	x9, [sp, #8]
; CHECK-CPA-O0:          mul	x10, x10, x9
; CHECK-CPA-O0:          add	x8, x8, x10
; CHECK-CPA-O0:          subs	x0, x8, x9
; CHECK-CPA-O0:          add	sp, sp, #32
; CHECK-CPA-O0:          ret
;
; CHECK-CPA-O3-LABEL:    arith2:
; CHECK-CPA-O3:          maddpt	x8, x9, x10, x0
; CHECK-CPA-O3:          ldr	x8, [x8, #24]
; CHECK-CPA-O3:          madd	x8, x1, x2, x8
; CHECK-CPA-O3:          sub	x0, x8, x2
; CHECK-CPA-O3:          add	sp, sp, #32
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  arith2:
; CHECK-NOCPA-O0:        mul	x9, x9, x10
; CHECK-NOCPA-O0:        add	x8, x8, x9
; CHECK-NOCPA-O0:        ldr	x8, [x8, #24]
; CHECK-NOCPA-O0:        ldr	x10, [sp, #16]
; CHECK-NOCPA-O0:        ldr	x9, [sp, #8]
; CHECK-NOCPA-O0:        mul	x10, x10, x9
; CHECK-NOCPA-O0:        add	x8, x8, x10
; CHECK-NOCPA-O0:        subs	x0, x8, x9
; CHECK-NOCPA-O0:        add	sp, sp, #32
; CHECK-NOCPA-O0:        ret
;
; CHECK-NOCPA-O3-LABEL:  arith2:
; CHECK-NOCPA-O3:        madd	x8, x8, x9, x0
; CHECK-NOCPA-O3:        ldr	x8, [x8, #24]
; CHECK-NOCPA-O3:        madd	x8, x1, x2, x8
; CHECK-NOCPA-O3:        sub	x0, x8, x2
; CHECK-NOCPA-O3:        add	sp, sp, #32
; CHECK-NOCPA-O3:        ret
entry:
  %5 = alloca ptr, align 8
  %6 = alloca i64, align 8
  %7 = alloca i64, align 8
  %8 = alloca i32, align 4
  store ptr %0, ptr %5, align 8
  store i64 %1, ptr %6, align 8
  store i64 %2, ptr %7, align 8
  store i32 %3, ptr %8, align 4
  %9 = load ptr, ptr %5, align 8
  %10 = load i32, ptr %8, align 4
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds %struct.my_type2, ptr %9, i64 %11
  %13 = getelementptr inbounds %struct.my_type2, ptr %12, i32 0, i32 3
  %14 = load i64, ptr %13, align 8
  %15 = load i64, ptr %6, align 8
  %16 = load i64, ptr %7, align 8
  %17 = mul nsw i64 %15, %16
  %18 = add nsw i64 %14, %17
  %19 = sub nsw i64 %18, %16
  ret i64 %19
}

@a = hidden global [2 x [1 x [2 x i8]]] [[1 x [2 x i8]] [[2 x i8] c"\01\01"], [1 x [2 x i8]] [[2 x i8] c"\01\01"]], align 1
@b = hidden global i16 0, align 2

define hidden void @multidim() {
; CHECK-CPA-O0-LABEL:    multidim:
; CHECK-CPA-O0:          adrp	x8, b
; CHECK-CPA-O0:          ldrh	w9, [x8, :lo12:b]
; CHECK-CPA-O0:          mov	w10, w9
; CHECK-CPA-O0:          ldrh	w8, [x8, :lo12:b]
; CHECK-CPA-O0:          add	w9, w8, #1
; CHECK-CPA-O0:          mov	w8, w9
; CHECK-CPA-O0:          sxtw	x9, w8
; CHECK-CPA-O0:          mov	w8, #2                          // =0x2
; CHECK-CPA-O0:          mov	w11, w8
; CHECK-CPA-O0:          adrp	x8, a
; CHECK-CPA-O0:          add	x8, x8, :lo12:a
; CHECK-CPA-O0:          addpt	x8, x8, x11
; CHECK-CPA-O0:          addpt	x8, x8, x10, lsl #1
; CHECK-CPA-O0:          addpt	x8, x8, x9
; CHECK-CPA-O0:          ldrb	w8, [x8]
;
; CHECK-CPA-O3-LABEL:    multidim:
; CHECK-CPA-O3:          ret
;
; CHECK-NOCPA-O0-LABEL:  multidim:
; CHECK-NOCPA-O0:        adrp	x8, b
; CHECK-NOCPA-O0:        ldrh	w9, [x8, :lo12:b]
; CHECK-NOCPA-O0:        mov	w10, w9
; CHECK-NOCPA-O0:        ldrh	w8, [x8, :lo12:b]
; CHECK-NOCPA-O0:        add	w9, w8, #1
; CHECK-NOCPA-O0:        adrp	x8, a
; CHECK-NOCPA-O0:        add	x8, x8, :lo12:a
; CHECK-NOCPA-O0:        add	x8, x8, #2
; CHECK-NOCPA-O0:        add	x8, x8, x10, lsl #1
; CHECK-NOCPA-O0:        add	x8, x8, w9, sxtw
; CHECK-NOCPA-O0:        ldrb	w8, [x8]
;
; CHECK-NOCPA-O3-LABEL:  multidim:
; CHECK-NOCPA-O3:        ret
entry:
  %0 = load i16, ptr @b, align 2
  %idxprom = zext i16 %0 to i64
  %arrayidx = getelementptr inbounds [1 x [2 x i8]], ptr getelementptr inbounds ([2 x [1 x [2 x i8]]], ptr @a, i64 0, i64 1), i64 0, i64 %idxprom
  %1 = load i16, ptr @b, align 2
  %conv = zext i16 %1 to i32
  %add = add nsw i32 %conv, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds [2 x i8], ptr %arrayidx, i64 0, i64 %idxprom1
  %2 = load i8, ptr %arrayidx2, align 1
  %tobool = icmp ne i8 %2, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:
  br label %if.end

if.end:
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

!6 = !{i64 0, i64 8, !7, i64 8, i64 8, !7, i64 16, i64 8, !7, i64 24, i64 8, !7, i64 32, i64 8, !7, i64 40, i64 8, !7}
!7 = !{!8, !8, i64 0}
!8 = !{!"long", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{i64 0, i64 8, !7, i64 8, i64 8, !7}
