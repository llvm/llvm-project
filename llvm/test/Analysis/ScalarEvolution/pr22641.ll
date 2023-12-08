; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

define i1 @main(i16 %a) {
; CHECK-LABEL: Classifying expressions for: @main
entry:
  br label %body

body:
  %dec2 = phi i16 [ %a, %entry ], [ %dec, %cond ]
  %dec = add i16 %dec2, -1
  %conv2 = zext i16 %dec2 to i32
  %conv = zext i16 %dec to i32
; CHECK:   %conv = zext i16 %dec to i32
; CHECK-NEXT: -->  {(zext i16 (-1 + %a) to i32),+,65535}<nuw><%body>
; CHECK-NOT:  -->  {(65535 + (zext i16 %a to i32)),+,65535}<nuw><%body>

  br label %cond

cond:
  br i1 false, label %body, label %exit

exit:
  %ret = icmp ne i32 %conv, 0
  ret i1 %ret
}
