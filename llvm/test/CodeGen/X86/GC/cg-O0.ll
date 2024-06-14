; RUN: llc -mtriple=x86_64 < %s -O0

define i32 @main() {
entry:
  call void @f()
  ret i32 0
}

define void @f() gc "ocaml" {
entry:
  %ptr.stackref = alloca ptr
  call void @llvm.gcroot(ptr %ptr.stackref, ptr null)
  ret void
}

declare void @llvm.gcroot(ptr, ptr) nounwind
