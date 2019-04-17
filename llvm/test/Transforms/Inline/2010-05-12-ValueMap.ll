; RUN: opt -inline -mergefunc -disable-output < %s

; This tests for a bug where the inliner kept the functions in a ValueMap after
; it had completed and a ModulePass started to run. LLVM would crash deleting
; a function that was still a key in the ValueMap.

define internal fastcc void @list_Cdr1918() nounwind inlinehint {
  unreachable
}

define internal fastcc void @list_PairSecond1927() nounwind inlinehint {
  call fastcc void @list_Cdr1918() nounwind inlinehint
  unreachable
}

define internal fastcc void @list_Cdr3164() nounwind inlinehint {
  unreachable
}

define internal fastcc void @list_Nconc3167() nounwind inlinehint {
  call fastcc void @list_Cdr3164() nounwind inlinehint
  unreachable
}

define void @term_Equal() nounwind {
  call fastcc void @list_Cdr3164() nounwind inlinehint
  unreachable
}
