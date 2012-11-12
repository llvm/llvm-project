; Test function attributes
; RUN: llvm-as < %s | llvm-dis | grep inline | count 2

define void @fn1() alwaysinline {
  ret void
}

define void @fn2() noinline {
  ret void
}

define void @fn3() {
  ret void
}
