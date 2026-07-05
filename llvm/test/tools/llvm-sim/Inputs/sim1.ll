define void @similar_func1() {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 2, ptr %a, align 4
  store i32 3, ptr %b, align 4
  store i32 4, ptr %c, align 4
  %al = load i32, ptr %a
  %bl = load i32, ptr %b
  %cl = load i32, ptr %c
  ret void
}

define void @similar_func2() {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 2, ptr %a, align 4
  store i32 3, ptr %b, align 4
  store i32 4, ptr %c, align 4
  %al = load i32, ptr %a
  %bl = load i32, ptr %b
  %cl = load i32, ptr %c
  ret void
}
