target triple = "powerpcle-unknown-linux-gnu"

@C32 = dso_local global i32 5, align 4
@undef_var32 = external dso_local global i32, align 4

define dso_local i32 @foo32(i32 %i) #0 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, ptr %i.addr, align 4
  %0 = load i32, ptr %i.addr, align 4
  %1 = load i32, ptr @undef_var32, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}
