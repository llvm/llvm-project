; Test that llvm-reduce can move def instructions down to uses.
;
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=sink-defs-to-uses --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK,RESULT %s < %t

declare i32 @leaf()
declare void @func()
declare void @use(i32)
declare void @use0(i32)
declare void @use1(i32)
declare void @use2(i32)
declare i32 @leaf_with_arg(i32)

; CHECK-LABEL: define i32 @sink_leaf_to_ret() {
; INTERESTING: call i32 @leaf()

; RESULT-NEXT: call void @func()
; RESULT-NEXT: %ret = call i32 @leaf()
; RESULT-NEXT: ret i32 %ret
define i32 @sink_leaf_to_ret() {
  %ret = call i32 @leaf()
  call void @func()
  ret i32 %ret
}

; CHECK-LABEL: define i32 @no_sink_leaf_to_ret() {
; INTERESTING: call i32 @leaf()
; INTERESTING: call void @func()

; RESULT: %ret = call i32 @leaf()
; RESULT-NEXT: call void @func()
; RESULT-NEXT: ret i32 %ret
define i32 @no_sink_leaf_to_ret() {
  %ret = call i32 @leaf()
  call void @func()
  ret i32 %ret
}

; CHECK-LABEL: define i32 @sink_across_trivial_block() {
; RESULT: {{^}}entry:
; RESULT-NEXT: br label %ret
; RESULT: {{^}}ret:
; RESULT-NEXT: call void @func
; RESULT-NEXT: %val = call i32 @leaf()
; RESULT-NEXT: ret i32 %val
define i32 @sink_across_trivial_block() {
entry:
  %val = call i32 @leaf()
  br label %ret

ret:
  call void @func()
  ret i32 %val
}

; CHECK-LABEL: define i32 @cannot_sink_phi_def(
; INTERESTING: phi i32

; RESULT: {{^}}b:
; RESULT-NEXT: %phi = phi i32
; RESULT-NEXT: call void @func(
; RESULT-NEXT: ret i32 %phi
define i32 @cannot_sink_phi_def(i1 %cond) {
entry:
  br i1 %cond, label %a, label %b

a:
  br label %b

b:
  %phi = phi i32 [ 0, %entry ], [ 1, %a ]
  call void @func()
  ret i32 %phi
}

; CHECK-LABEL: define i32 @cannot_sink_phi_use(
; INTERESTING: phi i32
define i32 @cannot_sink_phi_use(ptr %arg) {
entry:
  call void @func()
  br label %loop

loop:
  %phi = phi i32 [ 0, %entry ], [ %add, %loop ]
  call void @func()
  %def0 = call i32 @leaf()
  call void @func()
  %add = add i32 %phi, 1
  %loop.cond = load volatile i1, ptr %arg
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret i32 %phi
}

; CHECK-LABEL: define i32 @cannot_sink_past_other_use(
; INTERESTING: call i32 @leaf

; RESULT-NEXT: %val = call i32
; RESULT-NEXT: call void @use(i32 %val)
; RESULT-NEXT: ret i32 %val
define i32 @cannot_sink_past_other_use() {
  %val = call i32 @leaf()
  call void @use(i32 %val)
  ret i32 %val
}

; CHECK-LABEL: define void @no_sink_alloca(
; CHECK-NEXT: alloca
; RESULT-NEXT: call void @func
; RESULT-NEXT: store i32
; RESULT-NEXT: ret
define void @no_sink_alloca() {
  %alloca = alloca i32
  call void @func()
  store i32 0, ptr %alloca
  ret void
}

; CHECK-LABEL: define i32 @cannot_sink_callbr(
; CHECK: callbr i32

; RESULT: store i32 1
; RESULT-NEXT: ret i32 %load0

; RESULT: store i32 2
; RESULT-NEXT: ret i32 2

; RESULT: store i32 3
; RESULT-NEXT: ret i32 3
define i32 @cannot_sink_callbr(ptr %arg0, ptr %ptr1) {
entry:
  %load0 = load i32, ptr %arg0
  %callbr = callbr i32 asm "", "=r,r,!i,!i"(i32 %load0)
              to label %one [label %two, label %three]
one:
  store i32 1, ptr %ptr1
  ret i32 %load0

two:
  store i32 2, ptr %ptr1
  ret i32 2

three:
  store i32 3, ptr %ptr1
  ret i32 3
}

declare i32 @__gxx_personality_v0(...)
declare i32 @maybe_throwing_callee(i32)
declare void @did_not_throw(i32)

; landingpad must be first in the block, so it cannot be sunk.
; CHECK-LABEL: @cannot_sink_landingpad(
; INTERESTING: landingpad

; RESULT: %landing = landingpad { ptr, i32 }
; RESULT-NEXT: catch ptr
; RESULT-NEXT: call void @func(
; RESULT-NEXT: call void @func(
; RESULT-NEXT: extractvalue { ptr, i32 } %landing, 1
define void @cannot_sink_landingpad(i32 %arg) personality ptr @__gxx_personality_v0 {
bb:
  %i0 = invoke i32 @maybe_throwing_callee(i32 %arg)
          to label %bb3 unwind label %bb1

bb1:                                              ; preds = %bb
  %landing = landingpad { ptr, i32 }
          catch ptr null
  call void @func()
  call void @func()
  %extract0 = extractvalue { ptr, i32 } %landing, 1
  call void @use(i32 %extract0)
  br label %bb4

bb3:                                              ; preds = %bb
  call void @did_not_throw(i32 %i0)
  br label %bb4

bb4:                                              ; preds = %bb3, %bb1
  ret void
}

; CHECK-LABEL: define void @sink_multiple_uses() {
; INTERESTING: call i32 @leaf(
; INTERESTING: call void @use0(

; RESULT-NEXT: call void @func(
; RESULT-NEXT: %ret = call i32 @leaf()
; RESULT-NEXT: call void @use0(i32 %ret)
define void @sink_multiple_uses() {
  %ret = call i32 @leaf()
  call void @func()
  call void @use0(i32 %ret)
  call void @func()
  call void @use1(i32 %ret)
  call void @func()
  call void @use2(i32 %ret)
  ret void
}

; CHECK-LABEL: define i32 @can_sink_end_diamond(
; RESULT: entry:
; RESULT-NEXT: br i1

; RESULT: endif:
; RESULT-NEXT: %val = call i32 @leaf()
; RESULT-NEXT: call void @use(i32 %val)
; RESULT-NEXT: ret i32 %val
define i32 @can_sink_end_diamond(i1 %cond) {
entry:
  %val = call i32 @leaf()
  br i1 %cond, label %a, label %b

a:
  br label %endif

b:
  br label %endif

endif:
  call void @use(i32 %val)
  ret i32 %val
}

; CHECK-LABEL: define i32 @cannot_sink_diamond_end_0(
; RESULT: entry:
; RESULT-NEXT: %val = call i32 @leaf()
define i32 @cannot_sink_diamond_end_0(i1 %cond) {
entry:
  %val = call i32 @leaf()
  br i1 %cond, label %a, label %b

a:
  call void @use0(i32 %val)
  br label %endif

b:
  call void @use1(i32 %val)
  br label %endif

endif:
  ret i32 %val
}

; CHECK-LABEL: define void @cannot_sink_diamond_end_1(
; RESULT: entry:
; RESULT-NEXT: %val = call i32 @leaf()
define void @cannot_sink_diamond_end_1(i1 %cond) {
entry:
  %val = call i32 @leaf()
  br i1 %cond, label %a, label %b

a:
  call void @use0(i32 %val)
  br label %endif

b:
  call void @use1(i32 %val)
  br label %endif

endif:
  ret void
}
