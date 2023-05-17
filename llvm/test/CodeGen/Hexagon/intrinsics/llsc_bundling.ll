; RUN: llc -march=hexagon < %s
target triple = "hexagon-unknown--elf"

; Function Attrs: norecurse nounwind
define void @_Z4lockv() #0 {
entry:
  %__shared_owners = alloca i32, align 4
  %0 = cmpxchg weak ptr %__shared_owners, i32 0, i32 1 seq_cst seq_cst
  ret void
}

attributes #0 = { nounwind }
