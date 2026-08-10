@var5 = internal global i32 0, align 4
@var6 = internal global i32 0, align 4
@var7 = global ptr @var5, align 4
@var8 = global ptr @var6, align 4

define i32 @foo2() {
entry:
  %0 = load ptr, ptr @var7, align 4
  %1 = load i32, ptr %0, align 4
  %2 = load ptr, ptr @var8, align 4
  %3 = load i32, ptr %2, align 4
  %add = add nsw i32 %3, %1
  ret i32 %add
}
