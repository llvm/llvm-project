; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s --check-prefix=V7
; RUN: llc < %s -mtriple=armv8-none-linux-gnueabi | FileCheck %s -check-prefix=V8


define i32 @f(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: f:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp sgt i32 %a, %b
  %sub = sub nsw i32 %a, %b
  %sub. = select i1 %cmp, i32 %sub, i32 0
  ret i32 %sub.
}

define i32 @g(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: g:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp slt i32 %a, %b
  %sub = sub nsw i32 %b, %a
  %sub. = select i1 %cmp, i32 %sub, i32 0
  ret i32 %sub.
}

define i32 @h(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK-LABEL: h:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp sgt i32 %a, 3
  %sub = sub nsw i32 %a, 3
  %sub. = select i1 %cmp, i32 %sub, i32 %b
  ret i32 %sub.
}

; rdar://11725965
define i32 @i(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: i:
; CHECK: subs
; CHECK-NOT: cmp
  %cmp = icmp ult i32 %a, %b
  %sub = sub i32 %b, %a
  %sub. = select i1 %cmp, i32 %sub, i32 0
  ret i32 %sub.
}
; If CPSR is live-out, we can't remove cmp if there exists
; a swapped sub.
define i32 @j(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: j:
; CHECK: sub
; CHECK: cmp
  %cmp = icmp eq i32 %b, %a
  %sub = sub nsw i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %cmp2 = icmp sgt i32 %b, %a
  %sel = select i1 %cmp2, i32 %sub, i32 %a
  ret i32 %sel

if.else:
  ret i32 %sub
}

; If the sub/rsb instruction is predicated, we can't use the flags.
; <rdar://problem/12263428>
; Test case from MultiSource/Benchmarks/Ptrdist/bc/number.s
; CHECK: bc_raise
; CHECK: rsbeq
; CHECK: cmp
define i32 @bc_raise(i1 %cond) nounwind ssp {
entry:
  %val.2.i = select i1 %cond, i32 0, i32 undef
  %sub.i = sub nsw i32 0, %val.2.i
  %retval.0.i = select i1 %cond, i32 %val.2.i, i32 %sub.i
  %cmp1 = icmp eq i32 %retval.0.i, 0
  br i1 %cmp1, label %land.lhs.true, label %if.end11

land.lhs.true:                                    ; preds = %num2long.exit
  ret i32 17

if.end11:                                         ; preds = %num2long.exit
  ret i32 23
}

; When considering the producer of cmp's src as the subsuming instruction,
; only consider that when the comparison is to 0.
define i32 @cmp_src_nonzero(i32 %a, i32 %b, i32 %x, i32 %y) {
entry:
; CHECK-LABEL: cmp_src_nonzero:
; CHECK: sub
; CHECK: cmp
  %sub = sub i32 %a, %b
  %cmp = icmp eq i32 %sub, 17
  %ret = select i1 %cmp, i32 %x, i32 %y
  ret i32 %ret
}

define float @float_sel(i32 %a, i32 %b, float %x, float %y) {
entry:
; CHECK-LABEL: float_sel:
; CHECK-NOT: cmp
; V8-LABEL: float_sel:
; V8-NOT: cmp
; V8: vseleq.f32
  %sub = sub i32 %a, %b
  %cmp = icmp eq i32 %sub, 0
  %ret = select i1 %cmp, float %x, float %y
  ret float %ret
}

define double @double_sel(i32 %a, i32 %b, double %x, double %y) {
entry:
; CHECK-LABEL: double_sel:
; CHECK-NOT: cmp
; V8-LABEL: double_sel:
; V8-NOT: cmp
; V8: vseleq.f64
  %sub = sub i32 %a, %b
  %cmp = icmp eq i32 %sub, 0
  %ret = select i1 %cmp, double %x, double %y
  ret double %ret
}

@t = common global i32 0
define double @double_sub(i32 %a, i32 %b, double %x, double %y) {
entry:
; CHECK-LABEL: double_sub:
; CHECK: subs
; CHECK-NOT: cmp
; V8-LABEL: double_sub:
; V8: vsel
  %cmp = icmp sgt i32 %a, %b
  %sub = sub i32 %a, %b
  store i32 %sub, i32* @t
  %ret = select i1 %cmp, double %x, double %y
  ret double %ret
}

define double @double_sub_swap(i32 %a, i32 %b, double %x, double %y) {
entry:
; V7-LABEL: double_sub_swap:
; V7-NOT: cmp
; V7: subs
; V8-LABEL: double_sub_swap:
; V8-NOT: subs
; V8: cmp
; V8: vsel
  %cmp = icmp sgt i32 %a, %b
  %sub = sub i32 %b, %a
  %ret = select i1 %cmp, double %x, double %y
  store i32 %sub, i32* @t
  ret double %ret
}

declare void @abort()
declare void @exit(i32)

; If the comparison uses the V bit (signed overflow/underflow), we can't
; omit the comparison.
define i32 @cmp_slt0(i32 %a, i32 %b, i32 %x, i32 %y) {
entry:
; CHECK-LABEL: cmp_slt0
; CHECK: sub
; CHECK: cmn
; CHECK: bgt
  %load = load i32, i32* @t, align 4
  %sub = sub i32 %load, 17
  %cmp = icmp slt i32 %sub, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  call void @abort()
  unreachable

if.else:
  call void @exit(i32 0)
  unreachable
}

; Same for the C bit. (Note the ult X, 0 is trivially
; false, so the DAG combiner may or may not optimize it).
define i32 @cmp_ult0(i32 %a, i32 %b, i32 %x, i32 %y) {
entry:
; CHECK-LABEL: cmp_ult0
; CHECK: sub
; CHECK: cmp
; CHECK: bhs
  %load = load i32, i32* @t, align 4
  %sub = sub i32 %load, 17
  %cmp = icmp ult i32 %sub, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  call void @abort()
  unreachable

if.else:
  call void @exit(i32 0)
  unreachable
}
