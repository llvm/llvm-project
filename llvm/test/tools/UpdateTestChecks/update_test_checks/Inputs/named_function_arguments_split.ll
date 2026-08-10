; Check that we split named function arguments correctly into a separate CHECK line,
; ensuring the opening parenthesis is on the label name, avoiding incorrect label
; matches if function names are not prefix free.
;
; RUN: opt < %s -passes=instsimplify -S | FileCheck %s
;
define i32 @"foo"(i32 %named) {
entry:
  ret i32 %named
}
