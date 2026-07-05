; RUN: opt < %s -passes=inline,instcombine,reassociate,dse -disable-output
; PR12245

@a = common global i32 0, align 4
@d = common global i32 0, align 4

define i32 @fn2() nounwind uwtable ssp {
entry:
  %0 = load i32, ptr @a, align 4
  %dec = add nsw i32 %0, -1
  store i32 %dec, ptr @a, align 4
  %1 = load i32, ptr @d, align 4
  %sub = sub nsw i32 %dec, %1
  store i32 %sub, ptr @d, align 4
  %2 = load i32, ptr @a, align 4
  %dec1 = add nsw i32 %2, -1
  store i32 %dec1, ptr @a, align 4
  %3 = load i32, ptr @d, align 4
  %sub2 = sub nsw i32 %dec1, %3
  store i32 %sub2, ptr @d, align 4
  %4 = load i32, ptr @a, align 4
  %dec3 = add nsw i32 %4, -1
  store i32 %dec3, ptr @a, align 4
  %5 = load i32, ptr @d, align 4
  %sub4 = sub nsw i32 %dec3, %5
  store i32 %sub4, ptr @d, align 4
  %6 = load i32, ptr @a, align 4
  %dec5 = add nsw i32 %6, -1
  store i32 %dec5, ptr @a, align 4
  %7 = load i32, ptr @d, align 4
  %sub6 = sub nsw i32 %dec5, %7
  store i32 %sub6, ptr @d, align 4
  %8 = load i32, ptr @a, align 4
  %dec7 = add nsw i32 %8, -1
  store i32 %dec7, ptr @a, align 4
  %9 = load i32, ptr @d, align 4
  %sub8 = sub nsw i32 %dec7, %9
  store i32 %sub8, ptr @d, align 4
  ret i32 0
}

define i32 @fn1() nounwind uwtable ssp {
entry:
  %call = call i32 @fn2()
  ret i32 %call
}
