; RUN: opt < %s -passes=attributor

; Global variables that reference themselves alongside a function that is called indirectly
; used to cause an infinite loop in the attributor. The recursive reference was continually
; pushed back into the workload, causing the attributor to hang indefinitely.

@glob1 = global { ptr, ptr } { ptr @glob1, ptr @fnc1 }
@glob2 = global { ptr, ptr } { ptr @glob3, ptr @fnc2 }
@glob3 = global { ptr, ptr } { ptr @glob2, ptr @fnc2 }

define internal void @fnc1() {
  ret void
}

define internal void @fnc2() {
  ret void
}

define internal void @indr_caller(ptr %0) {
  call void %0()
  ret void
}

define void @main() {
  call void @indr_caller(ptr @fnc1)
  call void @indr_caller(ptr @fnc2)
  ret void
}
