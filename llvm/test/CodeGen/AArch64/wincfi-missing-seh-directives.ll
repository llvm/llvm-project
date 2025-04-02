; RUN: llc -mtriple=aarch64-windows %s --filetype obj -o /dev/null
; RUN: llc -mtriple=aarch64-windows %s --filetype asm -o - | FileCheck %s

; Check that it doesn't crash and that each instruction in the
; prologue has a corresponding seh directive.
;
; CHECK-NOT: error: Incorrect size for
; CHECK: foo:
; CHECK: .seh_proc foo
; CHECK: sub     sp, sp, #288
; CHECK: .seh_stackalloc 288
; CHECK: str     x19, [sp]                       // 8-byte Folded Spill
; CHECK: .seh_save_reg   x19, 0
; CHECK: str     x21, [sp, #8]                   // 8-byte Folded Spill
; CHECK: .seh_save_reg   x21, 8
; CHECK: stp     x23, x24, [sp, #16]             // 16-byte Folded Spill
; CHECK: .seh_save_regp  x23, 16
; CHECK: stp     x25, x26, [sp, #32]             // 16-byte Folded Spill
; CHECK: .seh_save_regp  x25, 32
; CHECK: stp     x27, x28, [sp, #48]             // 16-byte Folded Spill
; CHECK: .seh_save_regp  x27, 48
; CHECK: stp     x29, x30, [sp, #64]             // 16-byte Folded Spill
; CHECK: .seh_save_fplr  64
; CHECK: sub     sp, sp, #224
; CHECK: .seh_stackalloc 224
; CHECK: .seh_endprologue

target datalayout = "e-m:w-p:64:64-i32:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-windows-msvc19.42.34436"

%swift.refcounted = type { ptr, i64 }
%TScA_pSg = type <{ [16 x i8] }>
%T5repro4TestVSg = type <{ [32 x i8] }>
%T5repro4TestV = type <{ %TSS, %TSS }>
%TSS = type <{ %Ts11_StringGutsV }>
%Ts11_StringGutsV = type <{ %Ts13_StringObjectV }>
%Ts13_StringObjectV = type <{ %Ts6UInt64V, ptr }>
%Ts6UInt64V = type <{ i64 }>

declare swiftcc ptr @swift_task_alloc()

declare swifttailcc void @bar(ptr, ptr, i64, i64, i64, ptr, i64, i64, i64, i64, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr, i64, ptr)

define swifttailcc void @foo(ptr %0, ptr swiftasync %1, ptr swiftself %2, ptr %3, ptr %._guts2._object._object, ptr %.rid4._guts._object._object, ptr %4, ptr %.idx8, ptr %.idx8._guts._object._object, ptr %5, ptr %.rid9._guts._object._object, ptr %6) {
entry:
  %7 = load i64, ptr null, align 8
  %8 = load i64, ptr %3, align 8
  %9 = getelementptr <{ %swift.refcounted, %TScA_pSg, %TSS, %T5repro4TestVSg, %T5repro4TestV, %TSS, %TSS, %TSS, %T5repro4TestV, %TSS, %T5repro4TestV, %T5repro4TestV, %TSS }>, ptr %2, i32 0, i32 2
  %10 = load i64, ptr %9, align 8
  %11 = load ptr, ptr %1, align 8
  %12 = getelementptr <{ %swift.refcounted, %TScA_pSg, %TSS, %T5repro4TestVSg, %T5repro4TestV, %TSS, %TSS, %TSS, %T5repro4TestV, %TSS, %T5repro4TestV, %T5repro4TestV, %TSS }>, ptr %2, i32 0, i32 3
  %13 = load i64, ptr %.rid9._guts._object._object, align 8
  %14 = load i64, ptr %.idx8._guts._object._object, align 8
  %15 = load i64, ptr %5, align 8
  %16 = getelementptr { i64, i64, i64, i64 }, ptr %12, i32 0, i32 3
  %17 = load i64, ptr %16, align 8
  %18 = getelementptr <{ %swift.refcounted, %TScA_pSg, %TSS, %T5repro4TestVSg, %T5repro4TestV, %TSS, %TSS, %TSS, %T5repro4TestV, %TSS, %T5repro4TestV, %T5repro4TestV, %TSS }>, ptr %2, i32 0, i32 4
  %19 = load i64, ptr %18, align 8
  %.rid._guts._object._object = getelementptr %Ts13_StringObjectV, ptr %18, i32 0, i32 1
  %20 = load ptr, ptr %.rid._guts._object._object, align 8
  %21 = load i64, ptr %.rid4._guts._object._object, align 8
  %22 = load i64, ptr %0, align 8
  %23 = load ptr, ptr %6, align 8
  %24 = load i64, ptr %2, align 8
  %25 = load ptr, ptr %._guts2._object._object, align 8
  %26 = getelementptr <{ %swift.refcounted, %TScA_pSg, %TSS, %T5repro4TestVSg, %T5repro4TestV, %TSS, %TSS, %TSS, %T5repro4TestV, %TSS, %T5repro4TestV, %T5repro4TestV, %TSS }>, ptr %2, i32 0, i32 7
  %27 = load i64, ptr %26, align 8
  %._guts3._object._object = getelementptr %Ts13_StringObjectV, ptr %26, i32 0, i32 1
  %28 = load ptr, ptr %._guts3._object._object, align 8
  %29 = getelementptr <{ %swift.refcounted, %TScA_pSg, %TSS, %T5repro4TestVSg, %T5repro4TestV, %TSS, %TSS, %TSS, %T5repro4TestV, %TSS, %T5repro4TestV, %T5repro4TestV, %TSS }>, ptr %2, i32 0, i32 8
  %30 = load i64, ptr %29, align 8
  %.idx5 = getelementptr %T5repro4TestV, ptr %29, i32 0, i32 1
  %31 = load i64, ptr %.idx5, align 8
  %.idx5._guts._object._object = getelementptr %Ts13_StringObjectV, ptr %.idx5, i32 0, i32 1
  %32 = load ptr, ptr %.idx5._guts._object._object, align 8
  %33 = getelementptr <{ %swift.refcounted, %TScA_pSg, %TSS, %T5repro4TestVSg, %T5repro4TestV, %TSS, %TSS, %TSS, %T5repro4TestV, %TSS, %T5repro4TestV, %T5repro4TestV, %TSS }>, ptr %2, i32 0, i32 9
  %34 = load i64, ptr %33, align 8
  %35 = load i64, ptr %4, align 8
  %36 = load i64, ptr %.idx8, align 8
  %37 = load i64, ptr %1, align 8
  %38 = call swiftcc ptr @swift_task_alloc()
  store ptr null, ptr %3, align 8
  store ptr null, ptr %4, align 8
  musttail call swifttailcc void @bar(ptr null, ptr swiftasync %.rid4._guts._object._object, i64 %7, i64 %8, i64 %10, ptr %5, i64 %13, i64 %14, i64 %15, i64 %17, i64 %19, ptr %20, i64 %21, ptr %.idx8, i64 %22, ptr %23, i64 %24, ptr %25, i64 %27, ptr %28, i64 %30, ptr %.idx8._guts._object._object, i64 %31, ptr %32, i64 %34, ptr %._guts2._object._object, i64 %35, ptr %2, i64 %36, ptr %1, i64 %37, ptr %0, i64 0, ptr null, i64 0, ptr null)
  ret void
}
