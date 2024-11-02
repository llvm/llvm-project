; We should never separate alias from aliasee.
; RUN: llvm-split -j=3 -preserve-locals -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t2 | FileCheck --check-prefix=CHECK2 %s

; Checks are not critical here - verifier will assert if we fail.
; CHECK0: @funInternal2Alias = alias
; CHECK0: @funExternal2Alias = alias
; CHECK0: define internal i32 @funInternal2
; CHECK0: define i32 @funExternal2

; CHECK1: @funInternalAlias = alias
; CHECK1: define internal i32 @funInternal

; CHECK2: @funExternalAlias = alias
; CHECK2: define i32 @funExternal

@funInternalAlias = alias i32 (), ptr @funInternal
@funExternalAlias = alias i32 (), ptr @funExternal
@funInternal2Alias = alias i32 (), ptr @funInternal2
@funExternal2Alias = alias i32 (), ptr @funExternal2

define internal i32 @funInternal() {
entry:
  ret i32 0
}

define i32 @funExternal() {
entry:
  %x = call i32 @funInternalAlias()
  ret i32 %x
}

define internal i32 @funInternal2() {
entry:
  %x = call i32 @funInternalAlias()
  ret i32 %x
}

define i32 @funExternal2() {
entry:
  %x = call i32 @funInternal2()
  %y = call i32 @funExternalAlias()
  %z = add nsw i32 %x, %y
  ret i32 %z
}
