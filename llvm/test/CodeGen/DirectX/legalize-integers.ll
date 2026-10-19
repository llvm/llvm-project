; RUN: opt -S -passes=dxil-legalize -mtriple=dxil-pc-shadermodel6.3-library %s -o %t
; RUN: FileCheck %s < %t
; RUN: FileCheck %s --check-prefix=LEGAL < %t

; LEGAL-NOT: {{\bi(2|3|4|5|8|9|24|37|63)\b}}

define i1 @packed(<3 x i1> %v) {
; CHECK-LABEL: define i1 @packed(
; CHECK: extractelement <3 x i1> %v, i32 0
; CHECK: zext i1 {{.*}} to i32
; CHECK: shl i32 {{.*}}, 1
; CHECK: shl i32 {{.*}}, 2
; CHECK: icmp ne i32 {{.*}}, 0
  %p = bitcast <3 x i1> %v to i3
  %r = icmp ne i3 %p, 0
  ret i1 %r
}

define <3 x i1> @unpacked(i32 %v) {
; CHECK-LABEL: define <3 x i1> @unpacked(
; CHECK: trunc i32 %v to i1
; CHECK: lshr i32 %v, 1
; CHECK: lshr i32 %v, 2
; CHECK: ret <3 x i1>
  %n = trunc i32 %v to i3
  %r = bitcast i3 %n to <3 x i1>
  ret <3 x i1> %r
}

define i32 @wrap(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @wrap(
; CHECK: [[SUM:%.*]] = add i32 %a, %b
; CHECK-NEXT: [[MASK:%.*]] = and i32 [[SUM]], 7
; CHECK-NEXT: ret i32 [[MASK]]
  %x = trunc i32 %a to i3
  %y = trunc i32 %b to i3
  %sum = add nuw nsw i3 %x, %y
  %r = zext i3 %sum to i32
  ret i32 %r
}

define i1 @signed_cmp(i32 %v) {
; CHECK-LABEL: define i1 @signed_cmp(
; CHECK: [[SHL:%.*]] = shl i32 %v, 27
; CHECK-NEXT: [[SIGNED:%.*]] = ashr i32 [[SHL]], 27
; CHECK-NEXT: icmp slt i32 [[SIGNED]], -2
  %n = trunc i32 %v to i5
  %r = icmp slt i5 %n, -2
  ret i1 %r
}

define i64 @wide(i64 %a, i64 %b) {
; CHECK-LABEL: define i64 @wide(
; CHECK: [[MUL:%.*]] = mul i64 %a, %b
; CHECK-NEXT: [[MASK:%.*]] = and i64 [[MUL]], 137438953471
; CHECK-NEXT: ret i64 [[MASK]]
  %x = trunc i64 %a to i37
  %y = trunc i64 %b to i37
  %m = mul i37 %x, %y
  %r = zext i37 %m to i64
  ret i64 %r
}

define i32 @signed_div(i32 %a, i32 %b) {
; CHECK-LABEL: define i32 @signed_div(
; CHECK: shl i32 %a, 29
; CHECK: ashr i32 {{.*}}, 29
; CHECK: shl i32 %b, 29
; CHECK: ashr i32 {{.*}}, 29
; CHECK: sdiv i32
  %x = trunc i32 %a to i3
  %y = trunc i32 %b to i3
  %q = sdiv exact i3 %x, %y
  %r = sext i3 %q to i32
  ret i32 %r
}

define i32 @unsigned_shift(i32 %v, i32 %amount) {
; CHECK-LABEL: define i32 @unsigned_shift(
; CHECK: [[V:%.*]] = and i32 %v, 7
; CHECK-NEXT: [[A:%.*]] = and i32 %amount, 7
; CHECK-NEXT: [[R:%.*]] = lshr i32 [[V]], [[A]]
; CHECK-NEXT: ret i32 [[R]]
  %x = trunc i32 %v to i3
  %y = trunc i32 %amount to i3
  %s = lshr exact i3 %x, %y
  %r = zext i3 %s to i32
  ret i32 %r
}

define i32 @select(i1 %c, i32 %v) {
; CHECK-LABEL: define i32 @select(
; CHECK: [[S:%.*]] = select i1 %c, i32 %v, i32 7, !prof [[PROF:![0-9]+]]
; CHECK: and i32 [[S]], 7
  %n = trunc i32 %v to i3
  %s = select i1 %c, i3 %n, i3 -1, !prof !0
  %r = zext i3 %s to i32
  ret i32 %r
}

define float @signed_fp(i32 %v) {
; CHECK-LABEL: define float @signed_fp(
; CHECK: [[S:%.*]] = shl i32 %v, 29
; CHECK: [[A:%.*]] = ashr i32 [[S]], 29
; CHECK: sitofp i32 [[A]] to float
  %n = trunc i32 %v to i3
  %r = sitofp i3 %n to float
  ret float %r
}

define i32 @fp_to_unsigned(float %v) {
; CHECK-LABEL: define i32 @fp_to_unsigned(
; CHECK: [[F:%.*]] = fptoui float %v to i32
; CHECK-NEXT: [[M:%.*]] = and i32 [[F]], 7
; CHECK-NEXT: ret i32 [[M]]
  %n = fptoui float %v to i3
  %r = zext i3 %n to i32
  ret i32 %r
}

define i32 @freeze(i32 %v) {
; CHECK-LABEL: define i32 @freeze(
; CHECK: and i32 %v, 7
; CHECK-NOT: freeze
  %n = trunc i32 %v to i3
  %f = freeze i3 %n
  %r = zext i3 %f to i32
  ret i32 %r
}

define i32 @poison_select(i1 %c, i32 %v) {
; CHECK-LABEL: define i32 @poison_select(
; CHECK: [[S:%.*]] = select i1 %c, i32 poison, i32 %v
; CHECK-NEXT: [[R:%.*]] = and i32 [[S]], 7
; CHECK-NEXT: ret i32 [[R]]
  %n = trunc i32 %v to i3
  %s = select i1 %c, i3 poison, i3 %n
  %r = zext i3 %s to i32
  ret i32 %r
}

define i1 @preserve_i1(i16 %v) {
; CHECK-LABEL: define i1 @preserve_i1(
; CHECK-NEXT: [[R:%.*]] = trunc i16 %v to i1
; CHECK-NEXT: ret i1 [[R]]
  %r = trunc i16 %v to i1
  ret i1 %r
}

define i32 @i8_storage(i32 %v) {
; CHECK-LABEL: define i32 @i8_storage(
; CHECK: [[P:%.*]] = alloca i32
; CHECK: [[MASK:%.*]] = and i32 %v, 255
; CHECK-NEXT: store i32 [[MASK]], ptr [[P]]
; CHECK: [[LOAD:%.*]] = load i32, ptr [[P]]
; CHECK-NEXT: [[SHL:%.*]] = shl i32 [[LOAD]], 24
; CHECK-NEXT: [[SIGNED:%.*]] = ashr i32 [[SHL]], 24
; CHECK-NEXT: ret i32 [[SIGNED]]
  %p = alloca i8
  %n = trunc i32 %v to i8
  store i8 %n, ptr %p
  %l = load i8, ptr %p
  %r = sext i8 %l to i32
  ret i32 %r
}

define void @i8_store_argument(ptr %p, i32 %v) {
; CHECK-LABEL: define void @i8_store_argument(
; CHECK: [[MASK:%.*]] = and i32 %v, 255
; CHECK-NEXT: store i32 [[MASK]], ptr %p
; CHECK-NEXT: ret void
  %n = trunc i32 %v to i8
  store i8 %n, ptr %p
  ret void
}

define i32 @profiled_select(i1 %c, i32 %v) !prof !1 {
; CHECK-LABEL: define i32 @profiled_select(
; CHECK: select i1 %c, i32 %v, i32 0, !prof [[UNKNOWN:![0-9]+]]
  %n = trunc i32 %v to i3
  %s = select i1 %c, i3 %n, i3 0
  %r = zext i3 %s to i32
  ret i32 %r
}

define i32 @known_zero_bits(i32 %v) {
; CHECK-LABEL: define i32 @known_zero_bits(
; CHECK-NEXT: [[MASK:%.*]] = and i32 %v, 7
; CHECK-NEXT: ret i32 [[MASK]]
  %m = and i32 %v, 7
  %n = trunc i32 %m to i3
  %r = zext i3 %n to i32
  ret i32 %r
}

define i64 @known_sign_bits(i64 %v) {
; CHECK-LABEL: define i64 @known_sign_bits(
; CHECK-NEXT: [[S:%.*]] = ashr i64 %v, 27
; CHECK-NEXT: ret i64 [[S]]
  %s = ashr i64 %v, 27
  %n = trunc i64 %s to i37
  %r = sext i37 %n to i64
  ret i64 %r
}

define i32 @ptr_to_narrow(ptr %v) {
; CHECK-LABEL: define i32 @ptr_to_narrow(
; CHECK: [[P:%.*]] = ptrtoint ptr %v to i32
; CHECK-NEXT: [[M:%.*]] = and i32 [[P]], 7
; CHECK-NEXT: ret i32 [[M]]
  %n = ptrtoint ptr %v to i3
  %r = zext i3 %n to i32
  ret i32 %r
}

define ptr @narrow_to_ptr(i64 %v) {
; CHECK-LABEL: define ptr @narrow_to_ptr(
; CHECK: [[M:%.*]] = and i64 %v, 137438953471
; CHECK-NEXT: [[P:%.*]] = inttoptr i64 [[M]] to ptr
; CHECK-NEXT: ret ptr [[P]]
  %n = trunc i64 %v to i37
  %r = inttoptr i37 %n to ptr
  ret ptr %r
}

; CHECK: [[PROF]] = !{!"branch_weights", i32 3, i32 5}
; CHECK: [[UNKNOWN]] = !{!"unknown", !"dxil-legalize"}
!0 = !{!"branch_weights", i32 3, i32 5}
!1 = !{!"function_entry_count", i64 10}
