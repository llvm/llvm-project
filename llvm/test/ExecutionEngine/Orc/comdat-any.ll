; MachO doesn't support COMDATs
; UNSUPPORTED: system-darwin
;
; Works with MCJIT:
; RUN: lli --jit-kind=mcjit %s
;
; Fails assertion in ORC: Resolving symbol with incorrect flags
; RUN: lli --jit-kind=orc %s

$f = comdat any

define i32 @f() comdat {
entry:
  ret i32 0
}

define i32 @main() {
entry:
  %0 = call i32 @f()
  ret i32 %0
}
