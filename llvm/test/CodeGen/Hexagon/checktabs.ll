; RUN: llc -march=hexagon < %s | FileCheck --strict-whitespace %s
; Make sure we are emitting tabs as formatting.
; CHECK:	{
; CHECK-NEXT:		{{jump|r}}
define i32 @foobar(i32 %a, i32 %b) {
  %1 = add i32 %a, %b
  ret i32 %1
}
