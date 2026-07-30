target triple = "powerpc64le-unknown-linux-gnu"

@C64 = dso_local global i32 5, align 4
@static_var64 = internal global i32 2, align 4

define dso_local signext i32 @bar64() #0 {
entry:
  %0 = load i32, ptr @static_var64, align 4
  %1 = load i32, ptr @C64, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}
