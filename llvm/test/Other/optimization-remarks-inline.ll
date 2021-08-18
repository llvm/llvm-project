; RUN: opt < %s -inline -pass-remarks='inline' -S 2>&1 | FileCheck %s
; RUN: opt < %s -inline -pass-remarks='inl.*' -S 2>&1 | FileCheck %s
; RUN: opt < %s -inline -pass-remarks='vector' -pass-remarks='inl' -S 2>&1 | FileCheck %s

; These two should not yield an inline remark for the same reason.
; In the first command, we only ask for vectorizer remarks, in the
; second one we ask for the inliner, but we then ask for the vectorizer
; (thus overriding the first flag).
; RUN: opt < %s -inline -pass-remarks='vector' -S 2>&1 | FileCheck --check-prefix=REMARKS %s
; RUN: opt < %s -inline -pass-remarks='inl' -pass-remarks='vector' -S 2>&1 | FileCheck --check-prefix=REMARKS %s

; RUN: opt < %s -inline -S 2>&1 | FileCheck --check-prefix=REMARKS %s
; RUN: not --crash opt < %s -pass-remarks='(' 2>&1 | FileCheck --check-prefix=BAD-REGEXP %s

define i32 @foo(i32 %x, i32 %y) #0 {
entry:
  %x.addr = alloca i32, align 4
  %y.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  store i32 %y, i32* %y.addr, align 4
  %0 = load i32, i32* %x.addr, align 4
  %1 = load i32, i32* %y.addr, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

define i32 @bar(i32 %j) #0 {
entry:
  %j.addr = alloca i32, align 4
  store i32 %j, i32* %j.addr, align 4
  %0 = load i32, i32* %j.addr, align 4
  %1 = load i32, i32* %j.addr, align 4
  %sub = sub nsw i32 %1, 2
  %call = call i32 @foo(i32 %0, i32 %sub)
; CHECK: 'foo' inlined into 'bar'
; REMARKS-NOT: 'foo' inlined into 'bar'
  ret i32 %call
}

; BAD-REGEXP: Invalid regular expression '(' in -pass-remarks:
