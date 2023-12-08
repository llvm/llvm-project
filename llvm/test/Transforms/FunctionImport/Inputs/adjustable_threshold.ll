define void @globalfunc1() {
entry:
  call void @trampoline()
  ret void
}
; Adds an artificial level in the call graph to reduce the importing threshold
define void @trampoline() {
entry:
  call void @largefunction()
  ret void
}

define void @globalfunc2() {
entry:
  call void @largefunction()
  ret void
}


; Size is 5: if two layers below in the call graph the threshold will be 4,
; but if only one layer below the threshold will be 7.
define void @largefunction() {
  entry:
  call void @staticfunc2()
  call void @staticfunc2()
  call void @staticfunc2()
  call void @staticfunc2()
  call void @staticfunc2()
  ret void
}

define internal void @staticfunc2() {
entry:
  ret void
}


