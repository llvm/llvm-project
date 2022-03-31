; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

@arr = internal global [10 x i32] zeroinitializer, align 16

; CHECK: movl    %edx, arr(,%rdx,4)
define dso_local i32 @main() #0 {
entry:
  call void asm sideeffect inteldialect "mov dword ptr $0[rdx * $$4],edx", "=*m,~{dirflag},~{fpsr},~{flags}"([10 x i32]* elementtype([10 x i32]) @arr) #1, !srcloc !4
  ret i32 0
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 1}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{!"clang"}
!4 = !{i64 63}
