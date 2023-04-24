; RUN: opt -passes="ipsccp<func-spec>" -force-specialization -funcspec-for-literal-constant -funcspec-max-iters=3 -S < %s | FileCheck %s

define i64 @main() {
; CHECK:       define i64 @main
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[C1:%.*]] = call i64 @foo.1(i1 true, i64 3, i64 1)
; CHECK-NEXT:    [[C2:%.*]] = call i64 @foo.2(i1 false, i64 4, i64 -1)
; CHECK-NEXT:    ret i64 8
;
entry:
  %c1 = call i64 @foo(i1 true, i64 3, i64 1)
  %c2 = call i64 @foo(i1 false, i64 4, i64 -1)
  %add = add i64 %c1, %c2
  ret i64 %add
}

define internal i64 @foo(i1 %flag, i64 %m, i64 %n) {
;
; CHECK:       define internal i64 @foo.1
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %plus
; CHECK:       plus:
; CHECK-NEXT:    [[N0:%.*]] = call i64 @binop.4(i64 3, i64 1)
; CHECK-NEXT:    [[RES0:%.*]] = call i64 @bar.6(i64 4)
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    ret i64 undef
;
; CHECK:       define internal i64 @foo.2
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %minus
; CHECK:       minus:
; CHECK-NEXT:    [[N1:%.*]] = call i64 @binop.3(i64 4, i64 -1)
; CHECK-NEXT:    [[RES1:%.*]] = call i64 @bar.5(i64 3)
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    ret i64 undef
;
entry:
  br i1 %flag, label %plus, label %minus

plus:
  %n0 = call i64 @binop(i64 %m, i64 %n)
  %res0 = call i64 @bar(i64 %n0)
  br label %merge

minus:
  %n1 = call i64 @binop(i64 %m, i64 %n)
  %res1 = call i64 @bar(i64 %n1)
  br label %merge

merge:
  %res = phi i64 [ %res0, %plus ], [ %res1, %minus]
  ret i64 %res
}

define internal i64 @binop(i64 %x, i64 %y) {
;
; CHECK:       define internal i64 @binop.3
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 undef
;
; CHECK:       define internal i64 @binop.4
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret i64 undef
;
entry:
  %z = add i64 %x, %y
  ret i64 %z
}

define internal i64 @bar(i64 %n) {
;
; CHECK:       define internal i64 @bar.5
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %if.else
; CHECK:       if.else:
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    ret i64 undef
;
; CHECK:       define internal i64 @bar.6
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %if.then
; CHECK:       if.then:
; CHECK-NEXT:    br label %if.end
; CHECK:       if.end:
; CHECK-NEXT:    ret i64 undef
;
entry:
  %cmp = icmp sgt i64 %n, 3
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %res0 = sdiv i64 %n, 2
  br label %if.end

if.else:
  %res1 = mul i64 %n, 2
  br label %if.end

if.end:
  %res = phi i64 [ %res0, %if.then ], [ %res1, %if.else]
  ret i64 %res
}

