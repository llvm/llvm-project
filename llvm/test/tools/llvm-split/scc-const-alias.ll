; We should never separate alias from aliasee.
; RUN: llvm-split -j=3 -preserve-locals -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s

; Checks are not critical here - verifier will assert if we fail.
; CHECK0: @g1 = global i32 99
; CHECK0: @c1Alias = external global i8
; CHECK0: @g1Alias = internal alias i8, bitcast (i32* @g1 to i8*)

; CHECK1: @g1 = external global i32
; CHECK1: @c1Alias = internal alias i8, inttoptr (i64 42 to i8*)

; Third file is actually empty.
; CHECK2: @g1 = external global i32
; CHECK2: @g1Alias = external global i8
; CHECK2: @c1Alias = external global i8

@g1 = global i32 99

@g1Alias = internal alias i8, bitcast (i32* @g1 to i8*)
@c1Alias = internal alias i8, inttoptr (i64 42 to i8*)
@funExternalAlias = alias i8 (), i8 ()* @funExternal

define i8 @funExternal() {
entry:
  %x = load i8, i8* @g1Alias
  ret i8 %x
}

define i8 @funExternal2() {
entry:
  %x = load i8, i8* @c1Alias
  %y = call i8 @funExternalAlias()
  %z = add i8 %x, %y
  ret i8 %z
}
