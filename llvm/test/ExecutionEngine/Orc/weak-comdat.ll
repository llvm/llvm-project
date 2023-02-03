; RUN: lli -extra-module %p/Inputs/weak-comdat-def.ll %s

$c = comdat any

define weak i32 @f() comdat($c) {
entry:
  ret i32 0
}

define i32 @main() {
entry:
  %0 = call i32 @f()
  ret i32 %0
}
