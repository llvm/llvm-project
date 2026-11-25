; Check caller-aware stack-frame growth limits during inlining.
; RUN: opt < %s -passes=inline -inline-remark-attribute -S | FileCheck %s

declare void @use(ptr)

; Default behavior without attributes: large callee is inlined.
define i32 @caller_no_attr() {
; CHECK-LABEL: define i32 @caller_no_attr()
; CHECK-NOT: call i32 @callee_128_shared
  %r = call i32 @callee_128_shared()
  ret i32 %r
}

; Caller has 0 B stack, sole-callsite callee has 128 B (> 100 B): rejected.
define i32 @caller_zero_exceeds() #0 {
; CHECK-LABEL: define i32 @caller_zero_exceeds()
; CHECK: call i32 @callee_128_sole_zero() #[[REMARK_ATTR:[0-9]+]]
  %r = call i32 @callee_128_sole_zero()
  ret i32 %r
}

; Caller has 0 B stack, callee has 100 B (<= 100 B exact boundary): allowed.
define i32 @caller_zero_at_limit() #0 {
; CHECK-LABEL: define i32 @caller_zero_at_limit()
; CHECK-NOT: call i32 @callee_100
  %r = call i32 @callee_100()
  ret i32 %r
}

; Caller has 16 B stack, sole-callsite callee has 128 B (144 > 100, 800% > 40%): rejected.
define i32 @caller_small_exceeds() #0 {
; CHECK-LABEL: define i32 @caller_small_exceeds()
; CHECK: call i32 @callee_128_sole_small() #[[REMARK_ATTR]]
  %a = alloca [4 x i32], align 4
  call void @use(ptr %a)
  %r = call i32 @callee_128_sole_small()
  ret i32 %r
}

; Caller has 100 B stack, callee has 40 B (140 > 100, 40% <= 40% exact boundary): allowed.
define i32 @caller_growth_at_limit() #0 {
; CHECK-LABEL: define i32 @caller_growth_at_limit()
; CHECK-NOT: call i32 @callee_40
  %a = alloca [25 x i32], align 4
  call void @use(ptr %a)
  %r = call i32 @callee_40()
  ret i32 %r
}

; Caller has 100 B stack, callee has 44 B (144 > 100, 44% > 40%): rejected.
define i32 @caller_growth_exceeds() #0 {
; CHECK-LABEL: define i32 @caller_growth_exceeds()
; CHECK: call i32 @callee_44() #[[REMARK_ATTR]]
  %a = alloca [25 x i32], align 4
  call void @use(ptr %a)
  %r = call i32 @callee_44()
  ret i32 %r
}

; Caller already has 200 B (> 100 B), callee allocates 0 B: allowed.
define i32 @caller_large_callee_zero() #0 {
; CHECK-LABEL: define i32 @caller_large_callee_zero()
; CHECK-NOT: call i32 @callee_0
  %a = alloca [50 x i32], align 4
  call void @use(ptr %a)
  %r = call i32 @callee_0()
  ret i32 %r
}

; Cumulative stack growth across multiple callsites:
; First call (80 B) fits within 100 B and is inlined, growing caller to 80 B.
; Second call (40 B) makes combined 120 B > 100 B and growth 50% > 40%, so rejected.
define i32 @caller_cumulative() #0 {
; CHECK-LABEL: define i32 @caller_cumulative()
; CHECK-NOT: call i32 @callee_80
; CHECK: call i32 @callee_40() #[[REMARK_ATTR]]
  %r1 = call i32 @callee_80()
  %r2 = call i32 @callee_40()
  %s = add i32 %r1, %r2
  ret i32 %s
}

; alwaysinline callees are unaffected by stack-growth limits.
define i32 @caller_alwaysinline() #0 {
; CHECK-LABEL: define i32 @caller_alwaysinline()
; CHECK-NOT: call i32 @callee_512_alwaysinline
  %r = call i32 @callee_512_alwaysinline()
  ret i32 %r
}

; Only inline-large-stack-frame=100 attribute set (default growth = 1000%).
; Caller 16 B + callee 160 B = 176 B (> 100 B), growth = 1000% <= 1000%: allowed.
define i32 @caller_attr_frame_only_allow() #1 {
; CHECK-LABEL: define i32 @caller_attr_frame_only_allow()
; CHECK-NOT: call i32 @callee_160
  %a = alloca [4 x i32], align 4
  call void @use(ptr %a)
  %r = call i32 @callee_160()
  ret i32 %r
}

; Caller 16 B + callee 164 B = 180 B (> 100 B), growth = 1025% > 1000%: rejected.
define i32 @caller_attr_frame_only_reject() #1 {
; CHECK-LABEL: define i32 @caller_attr_frame_only_reject()
; CHECK: call i32 @callee_164() #[[REMARK_ATTR]]
  %a = alloca [4 x i32], align 4
  call void @use(ptr %a)
  %r = call i32 @callee_164()
  ret i32 %r
}

; Only inline-large-stack-frame-growth=40 attribute set (default frame = 256 B).
; Caller 0 B + callee 256 B = 256 B <= 256 B: allowed.
define i32 @caller_attr_growth_only_allow() #2 {
; CHECK-LABEL: define i32 @caller_attr_growth_only_allow()
; CHECK-NOT: call i32 @callee_256
  %r = call i32 @callee_256()
  ret i32 %r
}

; Caller 0 B + callee 260 B = 260 B > 256 B: rejected.
define i32 @caller_attr_growth_only_reject() #2 {
; CHECK-LABEL: define i32 @caller_attr_growth_only_reject()
; CHECK: call i32 @callee_260() #[[REMARK_ATTR]]
  %r = call i32 @callee_260()
  ret i32 %r
}

; Zero growth percentage rejects any positive stack allocation over the frame
; limit, while still allowing 0-byte callees.
define i32 @caller_attr_zero_growth() #3 {
; CHECK-LABEL: define i32 @caller_attr_zero_growth()
; CHECK-NOT: call i32 @callee_0
; CHECK: call i32 @callee_4() #[[REMARK_ATTR]]
  %a = alloca [25 x i32], align 4
  call void @use(ptr %a)
  %r1 = call i32 @callee_0()
  %r2 = call i32 @callee_4()
  %s = add i32 %r1, %r2
  ret i32 %s
}

; Very large growth percentage exercises the 128-bit overflow-safe comparison.
define i32 @caller_attr_huge_growth() #4 {
; CHECK-LABEL: define i32 @caller_attr_huge_growth()
; CHECK-NOT: call i32 @callee_128_shared
  %a = alloca [4 x i32], align 4
  call void @use(ptr %a)
  %r = call i32 @callee_128_shared()
  ret i32 %r
}

; Malformed/negative attribute values are ignored.
define i32 @caller_attr_invalid() #5 {
; CHECK-LABEL: define i32 @caller_attr_invalid()
; CHECK-NOT: call i32 @callee_128_shared
  %a = alloca [4 x i32], align 4
  call void @use(ptr %a)
  %r = call i32 @callee_128_shared()
  ret i32 %r
}

define internal i32 @callee_0() {
  ret i32 42
}

define internal i32 @callee_4() {
  %a = alloca i32, align 4
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_40() {
  %a = alloca [10 x i32], align 4
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_44() {
  %a = alloca [11 x i32], align 4
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_80() {
  %a = alloca [20 x i32], align 4
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_100() {
  %a = alloca [25 x i32], align 4
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

; Single-callsite internal functions to verify LastCallToStaticBonus does not
; override the stack-growth rejection.
define internal i32 @callee_128_sole_zero() {
  %a = alloca [32 x i32], align 16
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_128_sole_small() {
  %a = alloca [32 x i32], align 16
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_128_shared() {
  %a = alloca [32 x i32], align 16
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_160() {
  %a = alloca [40 x i32], align 4
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_164() {
  %a = alloca [41 x i32], align 4
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_256() {
  %a = alloca [64 x i32], align 4
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_260() {
  %a = alloca [65 x i32], align 4
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

define internal i32 @callee_512_alwaysinline() alwaysinline {
  %a = alloca [128 x i32], align 16
  call void @use(ptr %a)
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

; CHECK: attributes #[[REMARK_ATTR]] = { "inline-remark"="(cost=never): stacksize-growth" }
attributes #0 = { "inline-large-stack-frame"="100" "inline-large-stack-frame-growth"="40" }
attributes #1 = { "inline-large-stack-frame"="100" }
attributes #2 = { "inline-large-stack-frame-growth"="40" }
attributes #3 = { "inline-large-stack-frame"="100" "inline-large-stack-frame-growth"="0" }
attributes #4 = { "inline-large-stack-frame"="100" "inline-large-stack-frame-growth"="18446744073709551615" }
attributes #5 = { "inline-large-stack-frame"="-1" "inline-large-stack-frame-growth"="invalid" }
