; RUN: opt -basicaa -gvn -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

@x = common global i32 0, align 4
@y = common global i32 0, align 4

; GVN across unordered store (allowed)
define i32 @test1() nounwind uwtable ssp {
; CHECK-LABEL: test1
; CHECK: add i32 %x, %x
entry:
  %x = load i32, i32* @y
  store atomic i32 %x, i32* @x unordered, align 4
  %y = load i32, i32* @y
  %z = add i32 %x, %y
  ret i32 %z
}

; GVN across unordered load (allowed)
define i32 @test3() nounwind uwtable ssp {
; CHECK-LABEL: test3
; CHECK: add i32 %x, %x
entry:
  %x = load i32, i32* @y
  %y = load atomic i32, i32* @x unordered, align 4
  %z = load i32, i32* @y
  %a = add i32 %x, %z
  %b = add i32 %y, %a
  ret i32 %b
}

; GVN load to unordered load (allowed)
define i32 @test5() nounwind uwtable ssp {
; CHECK-LABEL: test5
; CHECK: add i32 %x, %x
entry:
  %x = load atomic i32, i32* @x unordered, align 4
  %y = load i32, i32* @x
  %z = add i32 %x, %y
  ret i32 %z
}

; GVN unordered load to load (unordered load must not be removed)
define i32 @test6() nounwind uwtable ssp {
; CHECK-LABEL: test6
; CHECK: load atomic i32, i32* @x unordered
entry:
  %x = load i32, i32* @x
  %x2 = load atomic i32, i32* @x unordered, align 4
  %x3 = add i32 %x, %x2
  ret i32 %x3
}

; GVN across release-acquire pair (forbidden)
define i32 @test7() nounwind uwtable ssp {
; CHECK-LABEL: test7
; CHECK: add i32 %x, %y
entry:
  %x = load i32, i32* @y
  store atomic i32 %x, i32* @x release, align 4
  %w = load atomic i32, i32* @x acquire, align 4
  %y = load i32, i32* @y
  %z = add i32 %x, %y
  ret i32 %z
}

; GVN across monotonic store (allowed)
define i32 @test9() nounwind uwtable ssp {
; CHECK-LABEL: test9
; CHECK: add i32 %x, %x
entry:
  %x = load i32, i32* @y
  store atomic i32 %x, i32* @x monotonic, align 4
  %y = load i32, i32* @y
  %z = add i32 %x, %y
  ret i32 %z
}

; GVN of an unordered across monotonic load (not allowed)
define i32 @test10() nounwind uwtable ssp {
; CHECK-LABEL: test10
; CHECK: add i32 %x, %y
entry:
  %x = load atomic i32, i32* @y unordered, align 4
  %clobber = load atomic i32, i32* @x monotonic, align 4
  %y = load atomic i32, i32* @y monotonic, align 4
  %z = add i32 %x, %y
  ret i32 %z
}

define i32 @PR22708(i1 %flag) {
; CHECK-LABEL: PR22708
entry:
  br i1 %flag, label %if.then, label %if.end

if.then:
  store i32 43, i32* @y, align 4
; CHECK: store i32 43, i32* @y, align 4
  br label %if.end

if.end:
  load atomic i32, i32* @x acquire, align 4
  %load = load i32, i32* @y, align 4
; CHECK: load atomic i32, i32* @x acquire, align 4
; CHECK: load i32, i32* @y, align 4
  ret i32 %load
}

; CHECK-LABEL: @test12(
; Can't remove a load over a ordering barrier
define i32 @test12(i1 %B, i32* %P1, i32* %P2) {
  %load0 = load i32, i32* %P1
  %1 = load atomic i32, i32* %P2 seq_cst, align 4
  %load1 = load i32, i32* %P1
  %sel = select i1 %B, i32 %load0, i32 %load1
  ret i32 %sel
  ; CHECK: load i32, i32* %P1
  ; CHECK: load i32, i32* %P1
}

; CHECK-LABEL: @test13(
; atomic to non-atomic forwarding is legal
define i32 @test13(i32* %P1) {
  %a = load atomic i32, i32* %P1 seq_cst, align 4
  %b = load i32, i32* %P1
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK: load atomic i32, i32* %P1
  ; CHECK: ret i32 0
}

; CHECK-LABEL: @test13b(
define i32 @test13b(i32* %P1) {
  store  atomic i32 0, i32* %P1 unordered, align 4
  %b = load i32, i32* %P1
  ret i32 %b
  ; CHECK: ret i32 0
}

; CHECK-LABEL: @test14(
; atomic to unordered atomic forwarding is legal
define i32 @test14(i32* %P1) {
  %a = load atomic i32, i32* %P1 seq_cst, align 4
  %b = load atomic i32, i32* %P1 unordered, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK: load atomic i32, i32* %P1 seq_cst
  ; CHECK-NEXT: ret i32 0
}

; CHECK-LABEL: @test15(
; implementation restriction: can't forward to stonger
; than unordered
define i32 @test15(i32* %P1, i32* %P2) {
  %a = load atomic i32, i32* %P1 seq_cst, align 4
  %b = load atomic i32, i32* %P1 seq_cst, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK: load atomic i32, i32* %P1
  ; CHECK: load atomic i32, i32* %P1
}

; CHECK-LABEL: @test16(
; forwarding non-atomic to atomic is wrong! (However,
; it would be legal to use the later value in place of the
; former in this particular example.  We just don't
; do that right now.)
define i32 @test16(i32* %P1, i32* %P2) {
  %a = load i32, i32* %P1, align 4
  %b = load atomic i32, i32* %P1 unordered, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK: load i32, i32* %P1
  ; CHECK: load atomic i32, i32* %P1
}

; CHECK-LABEL: @test16b(
define i32 @test16b(i32* %P1) {
  store i32 0, i32* %P1
  %b = load atomic i32, i32* %P1 unordered, align 4
  ret i32 %b
  ; CHECK: load atomic i32, i32* %P1
}

; Can't DSE across a full fence
define void @fence_seq_cst_store(i32* %P1, i32* %P2) {
; CHECK-LABEL: @fence_seq_cst_store(
; CHECK: store
; CHECK: store atomic
; CHECK: store
  store i32 0, i32* %P1, align 4
  store atomic i32 0, i32* %P2 seq_cst, align 4
  store i32 0, i32* %P1, align 4
  ret void
}

; Can't DSE across a full fence
define void @fence_seq_cst(i32* %P1, i32* %P2) {
; CHECK-LABEL: @fence_seq_cst(
; CHECK: store
; CHECK: fence seq_cst
; CHECK: store
  store i32 0, i32* %P1, align 4
  fence seq_cst
  store i32 0, i32* %P1, align 4
  ret void
}

; Can't DSE across a full singlethread fence
define void @fence_seq_cst_st(i32* %P1, i32* %P2) {
; CHECK-LABEL: @fence_seq_cst_st(
; CHECK: store
; CHECK: fence singlethread seq_cst
; CHECK: store
  store i32 0, i32* %P1, align 4
  fence singlethread seq_cst
  store i32 0, i32* %P1, align 4
  ret void
}

; Can't DSE across a full fence
define void @fence_asm_sideeffect(i32* %P1, i32* %P2) {
; CHECK-LABEL: @fence_asm_sideeffect(
; CHECK: store
; CHECK: call void asm sideeffect
; CHECK: store
  store i32 0, i32* %P1, align 4
  call void asm sideeffect "", ""()
  store i32 0, i32* %P1, align 4
  ret void
}

; Can't DSE across a full fence
define void @fence_asm_memory(i32* %P1, i32* %P2) {
; CHECK-LABEL: @fence_asm_memory(
; CHECK: store
; CHECK: call void asm
; CHECK: store
  store i32 0, i32* %P1, align 4
  call void asm "", "~{memory}"()
  store i32 0, i32* %P1, align 4
  ret void
}

; Can't remove a volatile load
define i32 @volatile_load(i32* %P1, i32* %P2) {
  %a = load i32, i32* %P1, align 4
  %b = load volatile i32, i32* %P1, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK-LABEL: @volatile_load(
  ; CHECK: load i32, i32* %P1
  ; CHECK: load volatile i32, i32* %P1
}

; Can't remove redundant volatile loads
define i32 @redundant_volatile_load(i32* %P1, i32* %P2) {
  %a = load volatile i32, i32* %P1, align 4
  %b = load volatile i32, i32* %P1, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK-LABEL: @redundant_volatile_load(
  ; CHECK: load volatile i32, i32* %P1
  ; CHECK: load volatile i32, i32* %P1
  ; CHECK: sub
}

; Can't DSE a volatile store
define void @volatile_store(i32* %P1, i32* %P2) {
; CHECK-LABEL: @volatile_store(
; CHECK: store volatile
; CHECK: store
  store volatile i32 0, i32* %P1, align 4
  store i32 3, i32* %P1, align 4
  ret void
}

; Can't DSE a redundant volatile store
define void @redundant_volatile_store(i32* %P1, i32* %P2) {
; CHECK-LABEL: @redundant_volatile_store(
; CHECK: store volatile
; CHECK: store volatile
  store volatile i32 0, i32* %P1, align 4
  store volatile i32 0, i32* %P1, align 4
  ret void
}

; Can value forward from volatiles
define i32 @test20(i32* %P1, i32* %P2) {
  %a = load volatile i32, i32* %P1, align 4
  %b = load i32, i32* %P1, align 4
  %res = sub i32 %a, %b
  ret i32 %res
  ; CHECK-LABEL: @test20(
  ; CHECK: load volatile i32, i32* %P1
  ; CHECK: ret i32 0
}

; We're currently conservative about widening
define i64 @widen1(i32* %P1) {
  ; CHECK-LABEL: @widen1(
  ; CHECK: load atomic i32, i32* %P1
  ; CHECK: load atomic i64, i64* %p2
  %p2 = bitcast i32* %P1 to i64*
  %a = load atomic i32, i32* %P1 unordered, align 4
  %b = load atomic i64, i64* %p2 unordered, align 4
  %a64 = sext i32 %a to i64
  %res = sub i64 %a64, %b
  ret i64 %res
}

; narrowing does work
define i64 @narrow(i32* %P1) {
  ; CHECK-LABEL: @narrow(
  ; CHECK: load atomic i64, i64* %p2
  ; CHECK-NOT: load atomic i32, i32* %P1
  %p2 = bitcast i32* %P1 to i64*
  %a64 = load atomic i64, i64* %p2 unordered, align 4
  %b = load atomic i32, i32* %P1 unordered, align 4
  %b64 = sext i32 %b to i64
  %res = sub i64 %a64, %b64
  ret i64 %res
}

; Missed optimization, we don't yet optimize ordered loads
define i64 @narrow2(i32* %P1) {
  ; CHECK-LABEL: @narrow2(
  ; CHECK: load atomic i64, i64* %p2
  ; CHECK: load atomic i32, i32* %P1
  %p2 = bitcast i32* %P1 to i64*
  %a64 = load atomic i64, i64* %p2 acquire, align 4
  %b = load atomic i32, i32* %P1 acquire, align 4
  %b64 = sext i32 %b to i64
  %res = sub i64 %a64, %b64
  ret i64 %res
}

; Note: The cross block FRE testing is deliberately light.  All of the tricky
; bits of legality are shared code with the block-local FRE above.  These
; are here only to show that we haven't obviously broken anything.

; unordered atomic to unordered atomic
define i32 @non_local_fre(i32* %P1) {
; CHECK-LABEL: @non_local_fre(
; CHECK: load atomic i32, i32* %P1
; CHECK: ret i32 0
; CHECK: ret i32 0
  %a = load atomic i32, i32* %P1 unordered, align 4
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %early, label %next
early:
  ret i32 %a
next:
  %b = load atomic i32, i32* %P1 unordered, align 4
  %res = sub i32 %a, %b
  ret i32 %res
}

; unordered atomic to non-atomic
define i32 @non_local_fre2(i32* %P1) {
; CHECK-LABEL: @non_local_fre2(
; CHECK: load atomic i32, i32* %P1
; CHECK: ret i32 0
; CHECK: ret i32 0
  %a = load atomic i32, i32* %P1 unordered, align 4
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %early, label %next
early:
  ret i32 %a
next:
  %b = load i32, i32* %P1
  %res = sub i32 %a, %b
  ret i32 %res
}

; Can't forward ordered atomics.
define i32 @non_local_fre3(i32* %P1) {
; CHECK-LABEL: @non_local_fre3(
; CHECK: load atomic i32, i32* %P1 acquire
; CHECK: ret i32 0
; CHECK: load atomic i32, i32* %P1 acquire
; CHECK: ret i32 %res
  %a = load atomic i32, i32* %P1 acquire, align 4
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %early, label %next
early:
  ret i32 %a
next:
  %b = load atomic i32, i32* %P1 acquire, align 4
  %res = sub i32 %a, %b
  ret i32 %res
}

declare void @clobber()

; unordered atomic to unordered atomic
define i32 @non_local_pre(i32* %P1) {
; CHECK-LABEL: @non_local_pre(
; CHECK: load atomic i32, i32* %P1 unordered
; CHECK: load atomic i32, i32* %P1 unordered
; CHECK: %b = phi i32 [ %b.pre, %early ], [ %a, %0 ]
; CHECK: ret i32 %b
  %a = load atomic i32, i32* %P1 unordered, align 4
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %early, label %next
early:
  call void @clobber()
  br label %next
next:
  %b = load atomic i32, i32* %P1 unordered, align 4
  ret i32 %b
}

; unordered atomic to non-atomic
define i32 @non_local_pre2(i32* %P1) {
; CHECK-LABEL: @non_local_pre2(
; CHECK: load atomic i32, i32* %P1 unordered
; CHECK: load i32, i32* %P1
; CHECK: %b = phi i32 [ %b.pre, %early ], [ %a, %0 ]
; CHECK: ret i32 %b
  %a = load atomic i32, i32* %P1 unordered, align 4
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %early, label %next
early:
  call void @clobber()
  br label %next
next:
  %b = load i32, i32* %P1
  ret i32 %b
}

; non-atomic to unordered atomic - can't forward!
define i32 @non_local_pre3(i32* %P1) {
; CHECK-LABEL: @non_local_pre3(
; CHECK: %a = load i32, i32* %P1
; CHECK: %b = load atomic i32, i32* %P1 unordered
; CHECK: ret i32 %b
  %a = load i32, i32* %P1
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %early, label %next
early:
  call void @clobber()
  br label %next
next:
  %b = load atomic i32, i32* %P1 unordered, align 4
  ret i32 %b
}

; ordered atomic to ordered atomic - can't forward
define i32 @non_local_pre4(i32* %P1) {
; CHECK-LABEL: @non_local_pre4(
; CHECK: %a = load atomic i32, i32* %P1 seq_cst
; CHECK: %b = load atomic i32, i32* %P1 seq_cst
; CHECK: ret i32 %b
  %a = load atomic i32, i32* %P1 seq_cst, align 4
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %early, label %next
early:
  call void @clobber()
  br label %next
next:
  %b = load atomic i32, i32* %P1 seq_cst, align 4
  ret i32 %b
}

; can't remove volatile on any path
define i32 @non_local_pre5(i32* %P1) {
; CHECK-LABEL: @non_local_pre5(
; CHECK: %a = load atomic i32, i32* %P1 seq_cst
; CHECK: %b = load volatile i32, i32* %P1
; CHECK: ret i32 %b
  %a = load atomic i32, i32* %P1 seq_cst, align 4
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %early, label %next
early:
  call void @clobber()
  br label %next
next:
  %b = load volatile i32, i32* %P1
  ret i32 %b
}


; ordered atomic to unordered atomic
define i32 @non_local_pre6(i32* %P1) {
; CHECK-LABEL: @non_local_pre6(
; CHECK: load atomic i32, i32* %P1 seq_cst
; CHECK: load atomic i32, i32* %P1 unordered
; CHECK: %b = phi i32 [ %b.pre, %early ], [ %a, %0 ]
; CHECK: ret i32 %b
  %a = load atomic i32, i32* %P1 seq_cst, align 4
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %early, label %next
early:
  call void @clobber()
  br label %next
next:
  %b = load atomic i32, i32* %P1 unordered, align 4
  ret i32 %b
}

