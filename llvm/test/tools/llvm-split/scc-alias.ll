; All of the functions in this module must end up
; in the same partition without change of scope.
; RUN: llvm-split -j=2 -preserve-locals -o %t %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=CHECK1 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=CHECK0 %s

; CHECK0: declare dso_local i32 @funInternal
; CHECK0: declare i32 @funExternal
; CHECK0: declare dso_local i32 @funInternal2
; CHECK0: declare i32 @funExternal2

; All functions are in the same file.
; Local functions are still local.
; CHECK1: define internal i32 @funInternal
; CHECK1: define i32 @funExternal
; CHECK1: define internal i32 @funInternal2
; CHECK1: define i32 @funExternal2


@funInternalAlias = internal alias i32 (), ptr @funInternal

define internal i32 @funInternal() {
entry:
  ret i32 0
}

; Direct call to local alias

define i32 @funExternal() {
entry:
  %x = call i32 @funInternalAlias()
  ret i32 %x
}

; Call to local function that calls local alias

define internal i32 @funInternal2() {
entry:
  %x = call i32 @funInternalAlias()
  ret i32 %x
}

define i32 @funExternal2() {
entry:
  %x = call i32 @funInternal2()
  ret i32 %x
}

