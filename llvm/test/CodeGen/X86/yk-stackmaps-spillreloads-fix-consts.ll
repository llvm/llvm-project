; RUN: llc -stop-after fix-stackmaps-spill-reloads --yk-insert-stackmaps --yk-stackmap-spillreloads-fix < %s  | FileCheck %s
; CHECK: STACKMAP 2, 0, 0, $rbp, -8, 3, implicit-def dead early-clobber $r11
@hash_search_j = dso_local global i64 0, align 8

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @luaH_getint(i64 noundef %0) #0 {
  %2 = alloca i64, align 8
  store i64 %0, i64* %2, align 8
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i64 @hash_search() #0 {
  %1 = alloca i64, align 8
  %2 = load i64, i64* @hash_search_j, align 8
  %3 = icmp ne i64 %2, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %0
  call void @luaH_getint(i64 noundef 9223372036854775807)
  store i64 9223372036854775807, i64* %1, align 8
  br label %6

5:                                                ; preds = %0
  store i64 0, i64* %1, align 8
  br label %6

6:                                                ; preds = %5, %4
  %7 = load i64, i64* %1, align 8
  ret i64 %7
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
