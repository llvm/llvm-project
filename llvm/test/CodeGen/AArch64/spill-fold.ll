; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu | FileCheck %s

declare i32 @bar()
declare i32 @baz()

; Check that the spill of the zero value gets stored directly instead
; of being copied from wzr and then stored.
define i32 @test_zr_spill_fold1(i1 %c) {
; CHECK-LABEL: test_zr_spill_fold1:
entry:
  br i1 %c, label %if.else, label %if.then

if.else:
; CHECK: bl bar
; CHECK-NEXT: str w0, [sp, #[[SLOT:[0-9]+]]]
  %call1 = tail call i32 @bar()
  br label %if.end

if.then:
; CHECK: bl baz
; CHECK-NEXT: str wzr, [sp, #[[SLOT]]]
  %call2 = tail call i32 @baz()
  br label %if.end

if.end:
  %x.0 = phi i32 [ 0, %if.then ], [ %call1, %if.else ]
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp}"() nounwind
  ret i32 %x.0
}

; Similar to test_zr_spill_fold1, but with mis-matched register
; class between %x.0 and the 0 from %if.then.
define i32 @test_zr_spill_fold2(i1 %c) {
; CHECK-LABEL: test_zr_spill_fold2:
entry:
  br i1 %c, label %if.else, label %if.then

if.else:
; CHECK: bl bar
; CHECK-NEXT: str w0, [sp, #[[SLOT:[0-9]+]]]
  %call1 = tail call i32 @bar()
  br label %if.end

if.then:
; CHECK: bl baz
; CHECK-NEXT: str wzr, [sp, #[[SLOT]]]
  %call2 = tail call i32 @baz()
  br label %if.end

if.end:
  %x.0 = phi i32 [ 0, %if.then ], [ %call1, %if.else ]
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp}"() nounwind
  %x.1 = add i32 %x.0, 1
  ret i32 %x.1
}

; Similar to test_zr_spill_fold1, but with a cross register-class copy feeding a spill store.
define float @test_cross_spill_fold(i32 %v) {
; CHECK-LABEL: test_cross_spill_fold:
entry:
; CHECK: str w0, [sp, #[[SLOT:[0-9]+]]]
  %v.f = bitcast i32 %v to float
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp},~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},~{s8},~{s9},~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19},~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29},~{s30},~{s31}"() nounwind
; CHECK: ldr s0, [sp, #[[SLOT]]]
  ret float %v.f
}

; Similar to test_cross_spill_fold, but with a cross register-class copy fed by a refill load.
define float @test_cross_spill_fold2(i32 %v) {
; CHECK-LABEL: test_cross_spill_fold2:
entry:
; CHECK: str w0, [sp, #[[SLOT:[0-9]+]]]
  call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp},~{lr},~{sp},~{s0},~{s1},~{s2},~{s3},~{s4},~{s5},~{s6},~{s7},~{s8},~{s9},~{s10},~{s11},~{s12},~{s13},~{s14},~{s15},~{s16},~{s17},~{s18},~{s19},~{s20},~{s21},~{s22},~{s23},~{s24},~{s25},~{s26},~{s27},~{s28},~{s29},~{s30},~{s31}"() nounwind
; CHECK: ldr s0, [sp, #[[SLOT]]]
  %v.f = bitcast i32 %v to float
  ret float %v.f
}

