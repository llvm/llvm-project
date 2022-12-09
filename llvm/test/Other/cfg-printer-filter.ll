; RUN: rm -f %t.other.dot %t-only.other.dot

;; Both f and func are dumped because their names contain the pattern 'f' as a substring.
; RUN: opt < %s -passes=dot-cfg -cfg-dot-filename-prefix=%t -cfg-func-name=f 2>/dev/null > /dev/null
; RUN: FileCheck %s -input-file=%t.f.dot -check-prefix=F
; RUN: FileCheck %s -input-file=%t.func.dot -check-prefix=Func
; RUN: not test -f %t.other.dot

; RUN: opt < %s -passes=dot-cfg-only -cfg-dot-filename-prefix=%t-only -cfg-func-name=f 2>/dev/null > /dev/null
; RUN: FileCheck %s -input-file=%t-only.f.dot -check-prefix=F
; RUN: FileCheck %s -input-file=%t-only.func.dot -check-prefix=Func
; RUN: not test -f %t-only.other.dot

; F: digraph "CFG for 'f' function"
define void @f(i32) {
entry:
  %check = icmp sgt i32 %0, 0
  br i1 %check, label %if, label %exit
if:                     ; preds = %entry
  br label %exit
exit:                   ; preds = %entry, %if
  ret void
}

; Func: digraph "CFG for 'func' function"
define void @func(i32) {
entry:
  %check = icmp sgt i32 %0, 0
  br label %exit
exit:                   ; preds = %entry
  ret void
}

define void @other(i32) {
entry:
  %check = icmp sgt i32 %0, 0
  br label %exit
exit:                   ; preds = %entry
  ret void
}
