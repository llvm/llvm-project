; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-linux-gnu | FileCheck %s

@var32 = global i32 0
@var64 = global i64 0

define void @test_zr() {
; CHECK-LABEL: test_zr:

  store i32 0, i32* @var32
; CHECK: str wzr, [{{x[0-9]+}}, {{#?}}:lo12:var32]
  store i64 0, i64* @var64
; CHECK: str xzr, [{{x[0-9]+}}, {{#?}}:lo12:var64]

  ret void
; CHECK: ret
}

define void @test_sp(i32 %val) {
; CHECK-LABEL: test_sp:

; Important correctness point here is that LLVM doesn't try to use xzr
; as an addressing register: "str w0, [xzr]" is not a valid A64
; instruction (0b11111 in the Rn field would mean "sp").
  %addr = getelementptr i32, i32* null, i64 0
  store i32 %val, i32* %addr
; CHECK: str {{w[0-9]+}}, [{{x[0-9]+|sp}}]

  ret void
; CHECK: ret
}

declare i32 @bar()
declare i32 @baz()

; Check that the spill of the zero value gets stored directly instead
; of being copied from wzr and then stored.
define i32 @test_zr_spill_copyprop1(i1 %c) {
; CHECK-LABEL: test_zr_spill_copyprop1:
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

; Similar to test_zr_spill_copyprop1, but with mis-matched register
; class between %x.0 and the 0 from %if.then.
define i32 @test_zr_spill_copyprop2(i1 %c) {
; CHECK-LABEL: test_zr_spill_copyprop2:
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
