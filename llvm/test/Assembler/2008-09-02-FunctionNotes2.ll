; Test function notes
; RUN: not llvm-as %s -o /dev/null -f |& grep "Attributes noinline alwaysinline are incompatible"
define void @fn1() alwaysinline  noinline {
  ret void
}

