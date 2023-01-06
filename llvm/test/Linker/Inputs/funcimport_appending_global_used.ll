@v = weak global i8 1
@llvm.used = appending global [2 x ptr] [ptr @foo, ptr @v]

define void @foo() {
  ret void
}
