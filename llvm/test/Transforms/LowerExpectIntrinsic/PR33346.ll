; RUN: opt -lower-expect -S < %s
; RUN: opt -passes='function(lower-expect)' -S < %s

define i64 @foo(i64 %arg) #0 {
bb:
  %tmp = alloca i64, align 8
  store i64 %arg, ptr %tmp, align 8
  %tmp1 = load i64, ptr %tmp, align 8
  %tmp2 = load i64, ptr %tmp, align 8
  %tmp3 = call i64 @llvm.expect.i64(i64 %tmp1, i64 %tmp2)
  ret i64 %tmp3
}

; Function Attrs: nounwind readnone
declare i64 @llvm.expect.i64(i64, i64)


!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 304723)"}
