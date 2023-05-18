; RUN: not llvm-as < %s > /dev/null 2>&1

void %test() {
  %X = add int 0, 1
  %X = add int 1, 2
  ret void
}
