; RUN: llc < %s -march=x86 -mcpu=pentiumpro -verify-machineinstrs | FileCheck %s

define i32 @f(i32 %X) {
entry:
; CHECK: f:
; CHECK: jns
	%tmp1 = add i32 %X, 1		; <i32> [#uses=1]
	%tmp = icmp slt i32 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tmp, label %cond_true, label %cond_next

cond_true:		; preds = %entry
	%tmp2 = tail call i32 (...)* @bar( )		; <i32> [#uses=0]
	br label %cond_next

cond_next:		; preds = %cond_true, %entry
	%tmp3 = tail call i32 (...)* @baz( )		; <i32> [#uses=0]
	ret i32 undef
}

declare i32 @bar(...)

declare i32 @baz(...)

; rdar://10633221
; rdar://11355268
define i32 @g(i32 %a, i32 %b) nounwind {
entry:
; CHECK: g:
; CHECK-NOT: test
; CHECK: cmovs
  %sub = sub nsw i32 %a, %b
  %cmp = icmp sgt i32 %sub, 0
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}

; rdar://10734411
define i32 @h(i32 %a, i32 %b) nounwind {
entry:
; CHECK: h:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp slt i32 %b, %a
  %sub = sub nsw i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
define i32 @i(i32 %a, i32 %b) nounwind {
entry:
; CHECK: i:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp sgt i32 %a, %b
  %sub = sub nsw i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
define i32 @j(i32 %a, i32 %b) nounwind {
entry:
; CHECK: j:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp ugt i32 %a, %b
  %sub = sub i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
define i32 @k(i32 %a, i32 %b) nounwind {
entry:
; CHECK: k:
; CHECK-NOT: cmp
; CHECK: cmov
; CHECK-NOT: movl
; CHECK: ret
  %cmp = icmp ult i32 %b, %a
  %sub = sub i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}
; redundant cmp instruction
define i32 @l(i32 %a, i32 %b) nounwind {
entry:
; CHECK: l:
; CHECK-NOT: cmp
  %cmp = icmp slt i32 %b, %a
  %sub = sub nsw i32 %a, %b
  %cond = select i1 %cmp, i32 %sub, i32 %a
  ret i32 %cond
}
define i32 @m(i32 %a, i32 %b) nounwind {
entry:
; CHECK: m:
; CHECK-NOT: cmp
  %cmp = icmp sgt i32 %a, %b
  %sub = sub nsw i32 %a, %b
  %cond = select i1 %cmp, i32 %b, i32 %sub
  ret i32 %cond
}
; If EFLAGS is live-out, we can't remove cmp if there exists
; a swapped sub.
define i32 @l2(i32 %a, i32 %b) nounwind {
entry:
; CHECK: l2:
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
define i32 @l3(i32 %a, i32 %b) nounwind {
entry:
; CHECK: l3:
; CHECK: sub
; CHECK-NOT: cmp
; CHECK: jge
  %cmp = icmp sgt i32 %b, %a
  %sub = sub nsw i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:
  ret i32 %sub

if.else:
  %add = add nsw i32 %sub, 1
  ret i32 %add
}
; rdar://11830760
; When Movr0 is between sub and cmp, we need to move "Movr0" before sub.
define i32 @l4(i32 %a, i32 %b) nounwind {
entry:
; CHECK: l4:
; CHECK: xor
; CHECK: sub
; CHECK-NOT: cmp
  %cmp = icmp sgt i32 %b, %a
  %sub = sub i32 %a, %b
  %.sub = select i1 %cmp, i32 0, i32 %sub
  ret i32 %.sub
}
; rdar://11540023
define i32 @n(i32 %x, i32 %y) nounwind {
entry:
; CHECK: n:
; CHECK-NOT: sub
; CHECK: cmp
  %sub = sub nsw i32 %x, %y
  %cmp = icmp slt i32 %sub, 0
  %y.x = select i1 %cmp, i32 %y, i32 %x
  ret i32 %y.x
}
; PR://13046
define void @o() nounwind uwtable {
entry:
  %0 = load i16* undef, align 2
  br i1 undef, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %entry
  unreachable

if.end.i:                                         ; preds = %entry
  br i1 undef, label %sw.bb, label %sw.default

sw.bb:                                            ; preds = %if.end.i
  br i1 undef, label %if.then44, label %if.end29

if.end29:                                         ; preds = %sw.bb
; CHECK: o:
; CHECK: cmp
  %1 = urem i16 %0, 10
  %cmp25 = icmp eq i16 %1, 0
  %. = select i1 %cmp25, i16 2, i16 0
  br i1 %cmp25, label %if.then44, label %sw.default

sw.default:                                       ; preds = %if.end29, %if.end.i
  br i1 undef, label %if.then.i96, label %if.else.i97

if.then.i96:                                      ; preds = %sw.default
  unreachable

if.else.i97:                                      ; preds = %sw.default
  unreachable

if.then44:                                        ; preds = %if.end29, %sw.bb
  %aModeRefSel.1.ph = phi i16 [ %., %if.end29 ], [ 3, %sw.bb ]
  br i1 undef, label %if.then.i103, label %if.else.i104

if.then.i103:                                     ; preds = %if.then44
  unreachable

if.else.i104:                                     ; preds = %if.then44
  ret void
}
; rdar://11855129
define i32 @p(i32 %a, i32 %b) nounwind {
entry:
; CHECK: p:
; CHECK-NOT: test
; CHECK: cmovs
  %add = add nsw i32 %b, %a
  %cmp = icmp sgt i32 %add, 0
  %add. = select i1 %cmp, i32 %add, i32 0
  ret i32 %add.
}
; PR13475
; If we have sub a, b and cmp b, a and the result of cmp is used
; by sbb, we should not optimize cmp away.
define i32 @q(i32 %j.4, i32 %w, i32 %el) {
; CHECK: q:
; CHECK: cmp
; CHECK-NEXT: sbb
  %tmp532 = add i32 %j.4, %w
  %tmp533 = icmp ugt i32 %tmp532, %el
  %tmp534 = icmp ult i32 %w, %el
  %or.cond = and i1 %tmp533, %tmp534
  %tmp535 = sub i32 %el, %w
  %j.5 = select i1 %or.cond, i32 %tmp535, i32 %j.4
  ret i32 %j.5
}
; rdar://11873276
define i8* @r(i8* %base, i32* nocapture %offset, i32 %size) nounwind {
entry:
; CHECK: r:
; CHECK: sub
; CHECK-NOT: cmp
; CHECK: j
; CHECK-NOT: sub
; CHECK: ret
  %0 = load i32* %offset, align 8
  %cmp = icmp slt i32 %0, %size
  br i1 %cmp, label %return, label %if.end

if.end:
  %sub = sub nsw i32 %0, %size
  store i32 %sub, i32* %offset, align 8
  %add.ptr = getelementptr inbounds i8* %base, i32 %sub
  br label %return

return:
  %retval.0 = phi i8* [ %add.ptr, %if.end ], [ null, %entry ]
  ret i8* %retval.0
}

; Test optimizations of dec/inc.
define i32 @dec(i32 %a) nounwind {
entry:
; CHECK: dec:
; CHECK: decl
; CHECK-NOT: test
; CHECK: cmovsl
  %sub = sub nsw i32 %a, 1
  %cmp = icmp sgt i32 %sub, 0
  %cond = select i1 %cmp, i32 %sub, i32 0
  ret i32 %cond
}

define i32 @inc(i32 %a) nounwind {
entry:
; CHECK: inc:
; CHECK: incl
; CHECK-NOT: test
; CHECK: cmovsl
  %add = add nsw i32 %a, 1
  %cmp = icmp sgt i32 %add, 0
  %cond = select i1 %cmp, i32 %add, i32 0
  ret i32 %cond
}

; PR13966
@b = common global i32 0, align 4
@a = common global i32 0, align 4
define i32 @test1(i32 %p1) nounwind uwtable {
entry:
; CHECK: test1:
; CHECK: testb
; CHECK: j
; CHECK: ret
  %0 = load i32* @b, align 4
  %cmp = icmp ult i32 %0, %p1
  %conv = zext i1 %cmp to i32
  %1 = load i32* @a, align 4
  %and = and i32 %conv, %1
  %conv1 = trunc i32 %and to i8
  %2 = urem i8 %conv1, 3
  %tobool = icmp eq i8 %2, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  %dec = add nsw i32 %1, -1
  store i32 %dec, i32* @a, align 4
  br label %if.end

if.end:
  ret i32 undef
}
