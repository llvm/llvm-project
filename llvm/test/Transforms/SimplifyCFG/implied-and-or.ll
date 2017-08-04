; RUN: opt %s -S -simplifycfg | FileCheck %s

declare void @foo()
declare void @bar()


; CHECK-LABEL: @test_and1
; CHECK: taken:
; CHECK-NOT: cmp3
; CHECK: call void @bar()
; CHECK-NEXT: call void @foo()
; CHECK: ret
define void @test_and1(i32 %a, i32 %b) {
entry:
  %cmp1 = icmp eq i32 %a, 0
  %cmp2 = icmp eq i32 %b, 0
  %and = and i1 %cmp1, %cmp2
  br i1 %and, label %taken, label %end

taken:
  call void @bar()
  %cmp3 = icmp eq i32 %a, 0  ;; <-- implied true
  br i1 %cmp3, label %if.then, label %end

if.then:
  call void @foo()
  br label %end

end:
  ret void
}

; We can't infer anything if the result of the 'and' is false
; CHECK-LABEL: @test_and2
; CHECK: taken:
; CHECK:   call void @bar()
; CHECK:   %cmp3
; CHECK:   br i1 %cmp3
; CHECK: if.then:
; CHECK:   call void @foo()
; CHECK: ret
define void @test_and2(i32 %a, i32 %b) {
entry:
  %cmp1 = icmp eq i32 %a, 0
  %cmp2 = icmp eq i32 %b, 0
  %and = and i1 %cmp1, %cmp2
  br i1 %and, label %end, label %taken

taken:
  call void @bar()
  %cmp3 = icmp eq i32 %a, 0
  br i1 %cmp3, label %if.then, label %end

if.then:
  call void @foo()
  br label %end

end:
  ret void
}

; CHECK-LABEL: @test_or1
; CHECK: taken:
; CHECK-NOT: cmp3
; CHECK: call void @bar()
; CHECK-NEXT: call void @foo()
; CHECK: ret
define void @test_or1(i32 %a, i32 %b) {
entry:
  %cmp1 = icmp eq i32 %a, 0
  %cmp2 = icmp eq i32 %b, 0
  %or = or i1 %cmp1, %cmp2
  br i1 %or, label %end, label %taken

taken:
  call void @bar()
  %cmp3 = icmp ne i32 %a, 0   ;; <-- implied true
  br i1 %cmp3, label %if.then, label %end

if.then:
  call void @foo()
  br label %end

end:
  ret void
}

; We can't infer anything if the result of the 'or' is true
; CHECK-LABEL: @test_or2
; CHECK:   call void @bar()
; CHECK:   %cmp3
; CHECK:   br i1 %cmp3
; CHECK: if.then:
; CHECK:   call void @foo()
; CHECK: ret
define void @test_or2(i32 %a, i32 %b) {
entry:
  %cmp1 = icmp eq i32 %a, 0
  %cmp2 = icmp eq i32 %b, 0
  %or = or i1 %cmp1, %cmp2
  br i1 %or, label %taken, label %end

taken:
  call void @bar()
  %cmp3 = icmp eq i32 %a, 0
  br i1 %cmp3, label %if.then, label %end

if.then:
  call void @foo()
  br label %end

end:
  ret void
}

; We can recurse a tree of 'and' or 'or's.
; CHECK-LABEL: @test_and_recurse1
; CHECK: taken:
; CHECK-NEXT:  call void @bar()
; CHECK-NEXT:  call void @foo()
; CHECK-NEXT:  br label %end
; CHECK: ret
define void @test_and_recurse1(i32 %a, i32 %b, i32 %c) {
entry:
  %cmpa = icmp eq i32 %a, 0
  %cmpb = icmp eq i32 %b, 0
  %cmpc = icmp eq i32 %c, 0
  %and1 = and i1 %cmpa, %cmpb
  %and2 = and i1 %and1, %cmpc
  br i1 %and2, label %taken, label %end

taken:
  call void @bar()
  %cmp3 = icmp eq i32 %a, 0
  br i1 %cmp3, label %if.then, label %end

if.then:
  call void @foo()
  br label %end

end:
  ret void
}

; Check to make sure we don't recurse too deep.
; CHECK-LABEL: @test_and_recurse2
; CHECK: taken:
; CHECK-NEXT:  call void @bar()
; CHECK-NEXT:  %cmp3 = icmp eq i32 %a, 0
; CHECK-NEXT:  br i1 %cmp3, label %if.then, label %end
; CHECK: ret
define void @test_and_recurse2(i32 %a, i32 %b, i32 %c, i32 %d, i32 %e, i32 %f,
                               i32 %g, i32 %h) {
entry:
  %cmpa = icmp eq i32 %a, 0
  %cmpb = icmp eq i32 %b, 0
  %cmpc = icmp eq i32 %c, 0
  %cmpd = icmp eq i32 %d, 0
  %cmpe = icmp eq i32 %e, 0
  %cmpf = icmp eq i32 %f, 0
  %cmpg = icmp eq i32 %g, 0
  %cmph = icmp eq i32 %h, 0
  %and1 = and i1 %cmpa, %cmpb
  %and2 = and i1 %and1, %cmpc
  %and3 = and i1 %and2, %cmpd
  %and4 = and i1 %and3, %cmpe
  %and5 = and i1 %and4, %cmpf
  %and6 = and i1 %and5, %cmpg
  %and7 = and i1 %and6, %cmph
  br i1 %and7, label %taken, label %end

taken:
  call void @bar()
  %cmp3 = icmp eq i32 %a, 0 ; <-- can be implied true
  br i1 %cmp3, label %if.then, label %end

if.then:
  call void @foo()
  br label %end

end:
  ret void
}
