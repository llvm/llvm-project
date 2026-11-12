; Test that, for a 64 bit signed rem, a libcall to allrem is made on Windows
; unless we have libgcc.

; RUN: llc < %s -mtriple i386-pc-win32 | FileCheck %s
; RUN: llc < %s -mtriple i386-pc-cygwin | FileCheck %s -check-prefix USEMODDI
; RUN: llc < %s -mtriple i386-pc-mingw32 | FileCheck %s -check-prefix USEMODDI
; PR10305
; END.

define i32 @main(i32 %argc, ptr nocapture %argv) nounwind readonly {
entry:
  %conv4 = sext i32 %argc to i64
  %div = srem i64 84, %conv4
  %conv7 = trunc i64 %div to i32
  ret i32 %conv7
}

; CHECK: allrem
; USEMODDI: moddi3
