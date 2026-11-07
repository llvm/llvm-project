; RUN: opt -passes="ipsccp<func-spec>,inline" -force-specialization -S  < %s | FileCheck %s --check-prefix=CHECK-ARGS

; Readonly and unspecified parameters
define i1 @recursiveFnA(i1 %flag, ptr readonly %p, ptr %q) {
entry:
  br i1 %flag, label %done, label %again

done:
  %same = icmp eq ptr %p, %q
  ret i1 %same

again:
   %r = call i1 @recursiveFnA(i1 true, ptr null, ptr null)
   ret i1 %r
}

define i1 @CallerA() {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %eq = call i1 @recursiveFnA(i1 true, ptr %a, ptr %a)
  ret i1 %eq
}

; CHECK-ARGS-LABEL: define i1 @recursiveFnA(i1 %flag, ptr readonly %p, ptr %q) {
; CHECK-ARGS: entry:
; CHECK-ARGS:   br i1 %flag, label %done, label %again
; CHECK-ARGS: done:
; CHECK-ARGS:  %same = icmp eq ptr %p, %q
; CHECK-ARGS:  ret i1 %same
; CHECK-ARGS: again:
; CHECK-ARGS:  ret i1 true
; CHECK-ARGS:}

; CHECK-ARGS-LABEL: define i1 @CallerA() {
; CHECK-ARGS: entry:
; CHECK-ARGS:  %a = alloca i32, align 4
; CHECK-ARGS:  store i32 0, ptr %a, align 4
; CHECK-ARGS:  ret i1 true
; CHECK-ARGS: }

; All readonly
define i1 @recursiveFnB(i1 %flag, ptr readonly %p, ptr readonly %q) {
entry:
  br i1 %flag, label %done, label %again

done:
  %same = icmp eq ptr %p, %q
  ret i1 %same

again:
  %r = call i1 @recursiveFnB(i1 true, ptr null, ptr null)
  ret i1 %r
}

define i1 @CallerB() {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %eq = call i1 @recursiveFnB(i1 true, ptr %a, ptr %a)
  ret i1 %eq
}

; CHECK-ARGS-LABEL: define i1 @recursiveFnB(i1 %flag, ptr readonly %p, ptr readonly %q) {
; CHECK-ARGS: entry:
; CHECK-ARGS:   br i1 %flag, label %done, label %again
; CHECK-ARGS: done:
; CHECK-ARGS:   %same = icmp eq ptr %p, %q
; CHECK-ARGS:   ret i1 %same
; CHECK-ARGS: again:
; CHECK-ARGS:   ret i1 true
; CHECK-ARGS: }

; CHECK-ARGS-LABEL: define i1 @CallerB() {
; CHECK-ARGS: entry:
; CHECK-ARGS:   %a = alloca i32, align 4
; CHECK-ARGS:   store i32 0, ptr %a, align 4
; CHECK-ARGS:   ret i1 true
; CHECK-ARGS: }

; Readnone and unspecified parameters
define i1 @recursiveFnC(i1 %flag, ptr readnone %p, ptr %q) {
entry:
  br i1 %flag, label %done, label %again

done:
  %same = icmp eq ptr %p, %q
  ret i1 %same

again:
   %r = call i1 @recursiveFnC(i1 true, ptr null, ptr null)
   ret i1 %r
}

define i1 @CallerC() {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %eq = call i1 @recursiveFnC(i1 true, ptr %a, ptr %a)
  ret i1 %eq
}

; CHECK-ARGS-LABEL: define i1 @recursiveFnC(i1 %flag, ptr readnone %p, ptr %q) {
; CHECK-ARGS: entry:
; CHECK-ARGS:   br i1 %flag, label %done, label %again
; CHECK-ARGS: done:
; CHECK-ARGS:  %same = icmp eq ptr %p, %q
; CHECK-ARGS:  ret i1 %same
; CHECK-ARGS: again:
; CHECK-ARGS:  ret i1 true
; CHECK-ARGS: }

; CHECK-ARGS-LABEL: define i1 @CallerC() {
; CHECK-ARGS: entry:
; CHECK-ARGS:  %a = alloca i32, align 4
; CHECK-ARGS:  store i32 0, ptr %a, align 4
; CHECK-ARGS:  ret i1 true
; CHECK-ARGS: }

; All readnone
define i1 @recursiveFnD(i1 %flag, ptr readnone %p, ptr readnone %q) {
entry:
  br i1 %flag, label %done, label %again

done:
  %same = icmp eq ptr %p, %q
  ret i1 %same

again:
  %r = call i1 @recursiveFnD(i1 true, ptr null, ptr null)
  ret i1 %r
}

define i1 @CallerD() {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %eq = call i1 @recursiveFnD(i1 true, ptr %a, ptr %a)
  ret i1 %eq
}

; CHECK-ARGS-LABEL: define i1 @recursiveFnD(i1 %flag, ptr readnone %p, ptr readnone %q) {
; CHECK-ARGS: entry:
; CHECK-ARGS:   br i1 %flag, label %done, label %again
; CHECK-ARGS: done:
; CHECK-ARGS:   %same = icmp eq ptr %p, %q
; CHECK-ARGS:   ret i1 %same
; CHECK-ARGS: again:
; CHECK-ARGS:   ret i1 true
; CHECK-ARGS: }

; CHECK-ARGS-LABEL: define i1 @CallerD() {
; CHECK-ARGS: entry:
; CHECK-ARGS:   %a = alloca i32, align 4
; CHECK-ARGS:   store i32 0, ptr %a, align 4
; CHECK-ARGS:   ret i1 true
; CHECK-ARGS: }
