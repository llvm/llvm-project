; RUN: opt -basicaa -gvn -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

; CHECK: @foo
; CHECK: entry.end_crit_edge:
; CHECK:   %n.pre = load i32* %q.phi.trans.insert
; CHECK: then:
; CHECK:   store i32 %z
; CHECK: end:
; CHECK:   %n = phi i32 [ %n.pre, %entry.end_crit_edge ], [ %z, %then ]
; CHECK:   ret i32 %n

@G = external global [100 x i32]
define i32 @foo(i32 %x, i32 %z) {
entry:
  %tobool = icmp eq i32 %z, 0
  br i1 %tobool, label %end, label %then

then:
  %i = sext i32 %x to i64
  %p = getelementptr [100 x i32]* @G, i64 0, i64 %i
  store i32 %z, i32* %p
  br label %end

end:
  %j = sext i32 %x to i64
  %q = getelementptr [100 x i32]* @G, i64 0, i64 %j
  %n = load i32* %q
  ret i32 %n
}
