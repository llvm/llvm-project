; RUN: %lli -jit-kind=mcjit -force-interpreter=true %s > /dev/null

; Check interpreter isn't crashing in handling target extension type.

@g = global target("spirv.Event") zeroinitializer, align 8

define i32 @main() {
  %event = alloca target("spirv.Event"), align 8
  store target("spirv.Event") zeroinitializer, ptr %event, align 8
  %e = load target("spirv.Event"), ptr %event, align 8

  store target("spirv.Event") poison, ptr @g, align 8
  %e2 = load target("spirv.Event"), ptr @g, align 8

  ret i32 0
}
