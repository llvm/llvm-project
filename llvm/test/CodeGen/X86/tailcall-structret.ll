; RUN: llc < %s -mtriple=i686-unknown-linux -tailcallopt | FileCheck %s
define fastcc { ptr, ptr} @init({ ptr, ptr}, i32) {
entry:
      %2 = tail call fastcc { ptr, ptr } @init({ ptr, ptr} %0, i32 %1)
      ret { ptr, ptr} %2
; CHECK: jmp init
}
