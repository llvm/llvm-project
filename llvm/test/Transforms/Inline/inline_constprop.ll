; RUN: opt < %s -passes=inline -inline-threshold=20 -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -inline-threshold=20 -S | FileCheck %s

define internal i32 @callee1(i32 %A, i32 %B) {
  %C = sdiv i32 %A, %B
  ret i32 %C
}

define i32 @caller1() {
; CHECK-LABEL: define i32 @caller1(
; CHECK-NEXT: ret i32 3

  %X = call i32 @callee1( i32 10, i32 3 )
  ret i32 %X
}

define i32 @caller2() {
; Check that we can constant-prop through instructions after inlining callee21
; to get constants in the inlined callsite to callee22.
; FIXME: Currently, the threshold is fixed at 20 because we don't perform
; *recursive* cost analysis to realize that the nested call site will definitely
; inline and be cheap. We should eventually do that and lower the threshold here
; to 1.
;
; CHECK-LABEL: @caller2(
; CHECK-NOT: call void @callee2
; CHECK: ret

  %x = call i32 @callee21(i32 42, i32 48)
  ret i32 %x
}

define i32 @callee21(i32 %x, i32 %y) {
  %sub = sub i32 %y, %x
  %result = call i32 @callee22(i32 %sub)
  ret i32 %result
}

declare ptr @getptr()

define i32 @callee22(i32 %x) {
  %icmp = icmp ugt i32 %x, 42
  br i1 %icmp, label %bb.true, label %bb.false
bb.true:
  ; This block musn't be counted in the inline cost.
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  %x4 = add i32 %x3, 1
  %x5 = add i32 %x4, 1
  %x6 = add i32 %x5, 1
  %x7 = add i32 %x6, 1
  %x8 = add i32 %x7, 1

  ret i32 %x8
bb.false:
  ret i32 %x
}

define i32 @caller3() {
; Check that even if the expensive path is hidden behind several basic blocks,
; it doesn't count toward the inline cost when constant-prop proves those paths
; dead.
;
; CHECK-LABEL: @caller3(
; CHECK-NOT: call
; CHECK: ret i32 6

entry:
  %x = call i32 @callee3(i32 42, i32 48)
  ret i32 %x
}

define i32 @callee3(i32 %x, i32 %y) {
  %sub = sub i32 %y, %x
  %icmp = icmp ugt i32 %sub, 42
  br i1 %icmp, label %bb.true, label %bb.false

bb.true:
  %icmp2 = icmp ult i32 %sub, 64
  br i1 %icmp2, label %bb.true.true, label %bb.true.false

bb.true.true:
  ; This block musn't be counted in the inline cost.
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  %x4 = add i32 %x3, 1
  %x5 = add i32 %x4, 1
  %x6 = add i32 %x5, 1
  %x7 = add i32 %x6, 1
  %x8 = add i32 %x7, 1
  br label %bb.merge

bb.true.false:
  ; This block musn't be counted in the inline cost.
  %y1 = add i32 %y, 1
  %y2 = add i32 %y1, 1
  %y3 = add i32 %y2, 1
  %y4 = add i32 %y3, 1
  %y5 = add i32 %y4, 1
  %y6 = add i32 %y5, 1
  %y7 = add i32 %y6, 1
  %y8 = add i32 %y7, 1
  br label %bb.merge

bb.merge:
  %result = phi i32 [ %x8, %bb.true.true ], [ %y8, %bb.true.false ]
  ret i32 %result

bb.false:
  ret i32 %sub
}

declare {i8, i1} @llvm.uadd.with.overflow.i8(i8 %a, i8 %b)

define i8 @caller4(i8 %z) {
; Check that we can constant fold through intrinsics such as the
; overflow-detecting arithmetic intrinsics. These are particularly important
; as they are used heavily in standard library code and generic C++ code where
; the arguments are oftent constant but complete generality is required.
;
; CHECK-LABEL: @caller4(
; CHECK-NOT: call
; CHECK: ret i8 -1

entry:
  %x = call i8 @callee4(i8 254, i8 14, i8 %z)
  ret i8 %x
}

define i8 @callee4(i8 %x, i8 %y, i8 %z) {
  %uadd = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 %x, i8 %y)
  %o = extractvalue {i8, i1} %uadd, 1
  br i1 %o, label %bb.true, label %bb.false

bb.true:
  ret i8 -1

bb.false:
  ; This block musn't be counted in the inline cost.
  %z1 = add i8 %z, 1
  %z2 = add i8 %z1, 1
  %z3 = add i8 %z2, 1
  %z4 = add i8 %z3, 1
  %z5 = add i8 %z4, 1
  %z6 = add i8 %z5, 1
  %z7 = add i8 %z6, 1
  %z8 = add i8 %z7, 1
  ret i8 %z8
}

define i64 @caller5(i64 %y) {
; Check that we can round trip constants through various kinds of casts etc w/o
; losing track of the constant prop in the inline cost analysis.
;
; CHECK-LABEL: @caller5(
; CHECK-NOT: call
; CHECK: ret i64 -1

entry:
  %x = call i64 @callee5(i64 42, i64 %y)
  ret i64 %x
}

define i64 @callee5(i64 %x, i64 %y) {
  %inttoptr = inttoptr i64 %x to ptr
  %ptrtoint = ptrtoint ptr %inttoptr to i64
  %trunc = trunc i64 %ptrtoint to i32
  %zext = zext i32 %trunc to i64
  %cmp = icmp eq i64 %zext, 42
  br i1 %cmp, label %bb.true, label %bb.false

bb.true:
  ret i64 -1

bb.false:
  ; This block musn't be counted in the inline cost.
  %y1 = add i64 %y, 1
  %y2 = add i64 %y1, 1
  %y3 = add i64 %y2, 1
  %y4 = add i64 %y3, 1
  %y5 = add i64 %y4, 1
  %y6 = add i64 %y5, 1
  %y7 = add i64 %y6, 1
  %y8 = add i64 %y7, 1
  ret i64 %y8
}

define float @caller6() {
; Check that we can constant-prop through fcmp instructions
;
; CHECK-LABEL: @caller6(
; CHECK-NOT: call
; CHECK: ret
  %x = call float @callee6(float 42.0)
  ret float %x
}

define float @callee6(float %x) {
  %icmp = fcmp ugt float %x, 42.0
  br i1 %icmp, label %bb.true, label %bb.false

bb.true:
  ; This block musn't be counted in the inline cost.
  %x1 = fadd float %x, 1.0
  %x2 = fadd float %x1, 1.0
  %x3 = fadd float %x2, 1.0
  %x4 = fadd float %x3, 1.0
  %x5 = fadd float %x4, 1.0
  %x6 = fadd float %x5, 1.0
  %x7 = fadd float %x6, 1.0
  %x8 = fadd float %x7, 1.0
  ret float %x8

bb.false:
  ret float %x
}



define i32 @PR13412.main() {
; This is a somewhat complicated three layer subprogram that was reported to
; compute the wrong value for a branch due to assuming that an argument
; mid-inline couldn't be equal to another pointer.
;
; After inlining, the branch should point directly to the exit block, not to
; the intermediate block.
; CHECK: @PR13412.main
; CHECK: br i1 true, label %[[TRUE_DEST:.*]], label %[[FALSE_DEST:.*]]
; CHECK: [[FALSE_DEST]]:
; CHECK-NEXT: call void @PR13412.fail()
; CHECK: [[TRUE_DEST]]:
; CHECK-NEXT: ret i32 0

entry:
  %i1 = alloca i64
  store i64 0, ptr %i1
  %call = call i1 @PR13412.first(ptr %i1, ptr %i1)
  br i1 %call, label %cond.end, label %cond.false

cond.false:
  call void @PR13412.fail()
  br label %cond.end

cond.end:
  ret i32 0
}

define internal i1 @PR13412.first(ptr %a, ptr %b) {
entry:
  %call = call ptr @PR13412.second(ptr %a, ptr %b)
  %cmp = icmp eq ptr %call, %b
  ret i1 %cmp
}

declare void @PR13412.fail()

define internal ptr @PR13412.second(ptr %a, ptr %b) {
entry:
  %sub.ptr.lhs.cast = ptrtoint ptr %b to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %a to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 2
  %cmp = icmp ugt i64 %sub.ptr.div, 1
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %0 = load i32, ptr %a
  %1 = load i32, ptr %b
  %cmp1 = icmp eq i32 %0, %1
  br i1 %cmp1, label %return, label %if.end3

if.end3:
  br label %return

return:
  %retval.0 = phi ptr [ %b, %if.end3 ], [ %a, %if.then ]
  ret ptr %retval.0
}

declare i32 @PR28802.external(i32 returned %p1)

define internal i32 @PR28802.callee() {
entry:
  br label %cont

cont:
  %0 = phi i32 [ 0, %entry ]
  %call = call i32 @PR28802.external(i32 %0)
  ret i32 %call
}

define i32 @PR28802() {
entry:
  %call = call i32 @PR28802.callee()
  ret i32 %call
}

; CHECK-LABEL: define i32 @PR28802(
; CHECK: %[[call:.*]] = call i32 @PR28802.external(i32 0)
; CHECK: ret i32 %[[call]]

define internal i32 @PR28848.callee(i32 %p2, i1 %c) {
entry:
  br i1 %c, label %cond.end, label %cond.true

cond.true:
  br label %cond.end

cond.end:
  %cond = phi i32 [ 0, %cond.true ], [ %p2, %entry ]
  %or = or i32 %cond, %p2
  ret i32 %or
}

define i32 @PR28848() {
entry:
  %call = call i32 @PR28848.callee(i32 0, i1 false)
  ret i32 %call
}
; CHECK-LABEL: define i32 @PR28848(
; CHECK: ret i32 0

define internal void @callee7(i16 %param1, i16 %param2) {
entry:
  br label %bb

bb:
  %phi = phi i16 [ %param2, %entry ]
  %add = add i16 %phi, %param1
  ret void
}

declare i16 @caller7.external(i16 returned)

define void @caller7() {
bb1:
  %call = call i16 @caller7.external(i16 1)
  call void @callee7(i16 0, i16 %call)
  ret void
}
; CHECK-LABEL: define void @caller7(
; CHECK: %call = call i16 @caller7.external(i16 1)
; CHECK-NEXT: ret void

define float @caller8(float %y) {
; Check that we can constant-prop through fneg instructions
;
; CHECK-LABEL: @caller8(
; CHECK-NOT: call
; CHECK: ret
  %x = call float @callee8(float -42.0, float %y)
  ret float %x
}

define float @callee8(float %x, float %y) {
  %neg = fneg float %x
  %icmp = fcmp ugt float %neg, 42.0
  br i1 %icmp, label %bb.true, label %bb.false

bb.true:
  ; This block musn't be counted in the inline cost.
  %y1 = fadd float %y, 1.0
  %y2 = fadd float %y1, 1.0
  %y3 = fadd float %y2, 1.0
  %y4 = fadd float %y3, 1.0
  %y5 = fadd float %y4, 1.0
  %y6 = fadd float %y5, 1.0
  %y7 = fadd float %y6, 1.0
  %y8 = fadd float %y7, 1.0
  ret float %y8

bb.false:
  ret float %x
}
