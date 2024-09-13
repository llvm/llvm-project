; RUN: opt < %s -mtriple=arm-arm-none-eabi -passes=globalopt -S | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "thumbv6m-arm-none-eabi"

%struct.P = type { i32, [13 x i8] }

; CHECK-NOT: [16 x i8]
@.str = private unnamed_addr constant [13 x i8] c"hello world\0A\00", align 1
@.str.1 = private unnamed_addr constant [4 x i8] c"%s\0A\00", align 1
@__ARM_use_no_argv = global i32 1, section ".ARM.use_no_argv", align 4
@llvm.used = appending global [1 x ptr] [ptr @__ARM_use_no_argv], section "llvm.metadata"

; Function Attrs: nounwind
define hidden i32 @main() local_unnamed_addr #0 {
entry:
  %p = alloca %struct.P, align 4
  call void @llvm.lifetime.start(i64 20, ptr nonnull %p) #2
  store i32 10, ptr %p, align 4, !tbaa !3
  %arraydecay = getelementptr inbounds %struct.P, ptr %p, i32 0, i32 1, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(ptr align 1 %arraydecay, ptr align 1 @.str, i32 13, i1 false)
  %puts = call i32 @puts(ptr %arraydecay)
  call void @llvm.lifetime.end(i64 20, ptr nonnull %p) #2
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start(i64, ptr nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end(i64, ptr nocapture) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1) #1

; Function Attrs: nounwind
declare i32 @puts(ptr nocapture readonly) #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denormal-fp-math"="preserve-sign" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cortex-m0" "target-features"="+strict-align" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{!"Component: ARM Compiler 6 devbuild Tool: armclang [devbuild]"}
!3 = !{!4, !5, i64 0}
!4 = !{!"P", !5, i64 0, !6, i64 4}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
