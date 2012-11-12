; RUN: opt < %s -inline -internalize -disable-output
define void @foo() nounwind {
  ret void
}

define void @main(...) nounwind {
  call void @foo()
  ret void
}


