; RUN: lli -extra-module %p/Inputs/weak-comdat-def.ll %s
; XFAIL: target={{.*}}-darwin{{.*}}

declare i32 @g()

$f = comdat nodeduplicate

define weak i32 @f() comdat {
entry:
  %0 = call i32 @g()
  ret i32 %0
}

define i32 @main() {
entry:
  %0 = call i32 @f()
  ret i32 %0
}
