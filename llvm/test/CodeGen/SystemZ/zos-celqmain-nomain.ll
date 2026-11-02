; A module without a definition of main gets no CELQMAIN.
;
; RUN: llc < %s -mtriple=s390x-ibm-zos | FileCheck %s

; CHECK-NOT: CELQMAIN
; CHECK-NOT: CELQBST

declare signext i32 @main()

define signext i32 @f() {
  %r = call signext i32 @main()
  ret i32 %r
}
