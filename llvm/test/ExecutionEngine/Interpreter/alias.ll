; RUN: %lli -jit-kind=mcjit -force-interpreter %s

define i32 @func() {
entry:
  ret i32 0
}

@alias = alias i32 (), i32 ()* @func

define i32 @main() {
entry:
  %call = call i32 @alias()
  ret i32 %call
}
