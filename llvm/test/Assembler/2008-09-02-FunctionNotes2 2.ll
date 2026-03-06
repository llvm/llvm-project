; Test function notes
; RUN: not llvm-as %s -o /dev/null 2>&1 | grep "Attributes 'noinline and alwaysinline' are incompatible"
define void @fn1() alwaysinline  noinline {
  ret void
}

