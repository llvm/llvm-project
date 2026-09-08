; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll
; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; The 'throws' attribute marks a function whose return value is accompanied by
; a success/failure discriminant (herbception / deterministic exceptions).

; CHECK: define { i64, i1 } @foo(i64 %x) #0
; CHECK: attributes #0 = { throws }

define { i64, i1 } @foo(i64 %x) #0 {
entry:
  %r.i = insertvalue { i64, i1 } poison, i64 %x, 0
  %r = insertvalue { i64, i1 } %r.i, i1 false, 1
  ret { i64, i1 } %r
}

define void @bar() #0 {
entry:
  %c = call { i64, i1 } @foo(i64 42)
  %ok = extractvalue { i64, i1 } %c, 1
  br i1 %ok, label %err, label %done

err:
  ret void

done:
  ret void
}

attributes #0 = { throws }
