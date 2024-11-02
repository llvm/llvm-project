; RUN: opt < %s -S -partial-inliner -skip-partial-inlining-cost-analysis=true | FileCheck %s


define i32 @callee_most(i32 %v) unnamed_addr  #0 #1 {
entry:
  %cmp = icmp sgt i32 %v, 2000
  br i1 %cmp, label %if.then, label %if.end

if.then:
  br label %if.then2

if.then2:
  %sub = sub i32 %v, 10
  br label %if.end

if.end:
  %v2 = phi i32 [ %v, %entry ], [ %sub, %if.then2 ]
  %add = add nsw i32 %v2, 200
  ret i32 %add
}

define i32 @callee_noinline(i32 %v) optnone noinline {
entry:
  %cmp = icmp sgt i32 %v, 2000
  br i1 %cmp, label %if.then, label %if.end

if.then:
  br label %if.then2

if.then2:
  %sub = sub i32 %v, 10
  br label %if.end

if.end:
  %v2 = phi i32 [ %v, %entry ], [ %sub, %if.then2 ]
  %add = add nsw i32 %v2, 200
  ret i32 %add
}

define i32 @callee_writeonly(i32 %v) writeonly ssp {
entry:
  %cmp = icmp sgt i32 %v, 2000
  br i1 %cmp, label %if.then, label %if.end

if.then:
  br label %if.then2

if.then2:
  %sub = sub i32 %v, 10
  br label %if.end

if.end:
  %v2 = phi i32 [ %v, %entry ], [ %sub, %if.then2 ]
  %add = add nsw i32 %v2, 200
  ret i32 %add
}
; CHECK-LABEL: @caller
; CHECK: call void @callee_most.2.if.then(i32 %v
; CHECK: call i32 @callee_noinline(i32 %v)
; CHECK: call void @callee_writeonly.1.if.then(i32 %v
define i32 @caller(i32 %v) ssp {
entry:
  %c1 = call i32 @callee_most(i32 %v)
  %c2 = call i32 @callee_noinline(i32 %v)
  %c3 = call i32 @callee_writeonly(i32 %v)
  ret i32 %c3
}

; CHECK: define internal void @callee_writeonly.1.if.then(i32 %v, ptr %sub.out) [[FN_ATTRS0:#[0-9]+]]
; CHECK: define internal void @callee_most.2.if.then(i32 %v, ptr %sub.out)  [[FN_ATTRS:#[0-9]+]]

; attributes to preserve
attributes #0 = {
  inlinehint minsize noduplicate noimplicitfloat norecurse noredzone nounwind
  nonlazybind optsize safestack sanitize_address sanitize_hwaddress sanitize_memory
  sanitize_thread ssp sspreq sspstrong uwtable "foo"="bar"
  "patchable-function"="prologue-short-redirect" "probe-stack"="_foo_guard" "stack-probe-size"="4096" }

; CHECK: attributes [[FN_ATTRS0]] = { ssp
; CHECK: attributes [[FN_ATTRS]] = { inlinehint minsize noduplicate noimplicitfloat norecurse noredzone nounwind nonlazybind optsize safestack sanitize_address sanitize_hwaddress sanitize_memory sanitize_thread ssp sspreq sspstrong uwtable "foo"="bar" "patchable-function"="prologue-short-redirect" "probe-stack"="_foo_guard" "stack-probe-size"="4096" }

; attributes to drop
attributes #1 = {
  alignstack=16 convergent inaccessiblememonly inaccessiblemem_or_argmemonly naked
  noreturn readonly argmemonly returns_twice speculatable "thunk"
}
