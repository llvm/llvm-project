; RUN: opt -simplifycfg -instcombine < %s -simplifycfg-merge-cond-stores=true -simplifycfg-merge-cond-stores-aggressively=false -phi-node-folding-threshold=2 -S | FileCheck %s

; CHECK-LABEL: @test_simple
; This test should succeed and end up if-converted.
; CHECK: icmp eq i32 %b, 0
; CHECK-NEXT: icmp ne i32 %a, 0
; CHECK-NEXT: xor i1 %x2, true
; CHECK-NEXT: %[[x:.*]] = or i1 %{{.*}}, %{{.*}}
; CHECK-NEXT: br i1 %[[x]]
; CHECK: store
; CHECK-NOT: store
; CHECK: ret
define void @test_simple(i32* %p, i32 %a, i32 %b) {
entry:
  %x1 = icmp eq i32 %a, 0
  br i1 %x1, label %fallthrough, label %yes1

yes1:
  store i32 0, i32* %p
  br label %fallthrough

fallthrough:
  %x2 = icmp eq i32 %b, 0
  br i1 %x2, label %end, label %yes2

yes2:
  store i32 1, i32* %p
  br label %end

end:
  ret void
}

; CHECK-LABEL: @test_recursive
; This test should entirely fold away, leaving one large basic block.
; CHECK: store
; CHECK-NOT: store
; CHECK: ret
define void @test_recursive(i32* %p, i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %x1 = icmp eq i32 %a, 0
  br i1 %x1, label %fallthrough, label %yes1

yes1:
  store i32 0, i32* %p
  br label %fallthrough

fallthrough:
  %x2 = icmp eq i32 %b, 0
  br i1 %x2, label %next, label %yes2

yes2:
  store i32 1, i32* %p
  br label %next

next:
  %x3 = icmp eq i32 %c, 0
  br i1 %x3, label %fallthrough2, label %yes3

yes3:
  store i32 2, i32* %p
  br label %fallthrough2

fallthrough2:
  %x4 = icmp eq i32 %d, 0
  br i1 %x4, label %end, label %yes4

yes4:
  store i32 3, i32* %p
  br label %end


end:
  ret void
}

; CHECK-LABEL: @test_not_ifconverted
; The code in each diamond is too large - it won't be if-converted so our
; heuristics should say no.
; CHECK: store
; CHECK: store
; CHECK: ret
define void @test_not_ifconverted(i32* %p, i32 %a, i32 %b) {
entry:
  %x1 = icmp eq i32 %a, 0
  br i1 %x1, label %fallthrough, label %yes1

yes1:
  %y1 = or i32 %b, 55
  %y2 = add i32 %y1, 24
  %y3 = and i32 %y2, 67
  store i32 %y3, i32* %p
  br label %fallthrough

fallthrough:
  %x2 = icmp eq i32 %b, 0
  br i1 %x2, label %end, label %yes2

yes2:
  %z1 = or i32 %a, 55
  %z2 = add i32 %z1, 24
  %z3 = and i32 %z2, 67
  store i32 %z3, i32* %p
  br label %end

end:
  ret void
}

; CHECK-LABEL: @test_aliasing1
; The store to %p clobbers the previous store, so if-converting this would
; be illegal.
; CHECK: store
; CHECK: store
; CHECK: ret
define void @test_aliasing1(i32* %p, i32 %a, i32 %b) {
entry:
  %x1 = icmp eq i32 %a, 0
  br i1 %x1, label %fallthrough, label %yes1

yes1:
  store i32 0, i32* %p
  br label %fallthrough

fallthrough:
  %y1 = load i32, i32* %p
  %x2 = icmp eq i32 %y1, 0
  br i1 %x2, label %end, label %yes2

yes2:
  store i32 1, i32* %p
  br label %end

end:
  ret void
}

; CHECK-LABEL: @test_aliasing2
; The load from %q aliases with %p, so if-converting this would be illegal.
; CHECK: store
; CHECK: store
; CHECK: ret
define void @test_aliasing2(i32* %p, i32* %q, i32 %a, i32 %b) {
entry:
  %x1 = icmp eq i32 %a, 0
  br i1 %x1, label %fallthrough, label %yes1

yes1:
  store i32 0, i32* %p
  br label %fallthrough

fallthrough:
  %y1 = load i32, i32* %q
  %x2 = icmp eq i32 %y1, 0
  br i1 %x2, label %end, label %yes2

yes2:
  store i32 1, i32* %p
  br label %end

end:
  ret void
}

declare void @f()

; CHECK-LABEL: @test_diamond_simple
; This should get if-converted.
; CHECK: store
; CHECK-NOT: store
; CHECK: ret
define i32 @test_diamond_simple(i32* %p, i32* %q, i32 %a, i32 %b) {
entry:
  %x1 = icmp eq i32 %a, 0
  br i1 %x1, label %no1, label %yes1

yes1:
  store i32 0, i32* %p
  br label %fallthrough

no1:
  %z1 = add i32 %a, %b
  br label %fallthrough

fallthrough:
  %z2 = phi i32 [ %z1, %no1 ], [ 0, %yes1 ]
  %x2 = icmp eq i32 %b, 0
  br i1 %x2, label %no2, label %yes2

yes2:
  store i32 1, i32* %p
  br label %end

no2:
  %z3 = sub i32 %z2, %b
  br label %end

end:
  %z4 = phi i32 [ %z3, %no2 ], [ 3, %yes2 ]
  ret i32 %z4
}

; CHECK-LABEL: @test_diamond_alias3
; Now there is a call to f() in the bottom branch. The store in the first
; branch would now be reordered with respect to the call if we if-converted,
; so we must not.
; CHECK: store
; CHECK: store
; CHECK: ret
define i32 @test_diamond_alias3(i32* %p, i32* %q, i32 %a, i32 %b) {
entry:
  %x1 = icmp eq i32 %a, 0
  br i1 %x1, label %no1, label %yes1

yes1:
  store i32 0, i32* %p
  br label %fallthrough

no1:
  call void @f()
  %z1 = add i32 %a, %b
  br label %fallthrough

fallthrough:
  %z2 = phi i32 [ %z1, %no1 ], [ 0, %yes1 ]
  %x2 = icmp eq i32 %b, 0
  br i1 %x2, label %no2, label %yes2

yes2:
  store i32 1, i32* %p
  br label %end

no2:
  call void @f()
  %z3 = sub i32 %z2, %b
  br label %end

end:
  %z4 = phi i32 [ %z3, %no2 ], [ 3, %yes2 ]
  ret i32 %z4
}
