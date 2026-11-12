; REQUIRES: aarch64-registered-target

; RUN: opt -passes=lowertypetests %s -o %t.o
; RUN: llvm-dis %t.o -o - | FileCheck %s --check-prefix=CHECK-foobar
; CHECK-foobar: {{llvm.global.annotations = .*[foo|bar], .*[foo|bar],}}
; RUN: llvm-dis %t.o -o - | FileCheck %s --check-prefix=CHECK-cfi
; CHECK-cfi-NOT: {{llvm.global.annotations = .*cfi.*}}

target triple = "aarch64-none-linux-gnu"

@.src = private unnamed_addr constant [7 x i8] c"test.c\00", align 1
@.str = private unnamed_addr constant [30 x i8] c"annotation_string_literal_bar\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [7 x i8] c"test.c\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [30 x i8] c"annotation_string_literal_foo\00", section "llvm.metadata"
@llvm.global.annotations = appending global [2 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr @bar, ptr @.str, ptr @.str.1, i32 2, ptr null }, { ptr, ptr, ptr, i32, ptr } { ptr @foo, ptr @.str.2, ptr @.str.1, i32 1, ptr null }], section "llvm.metadata"

define i32 @bar(i32 noundef %0) #0 !type !8 !type !9 {
  %2 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = call i32 @foo(i32 noundef %3)
  ret i32 %4
}

declare !type !8 !type !9 i32 @foo(i32 noundef) #1

define i32 @test(i32 noundef %0) #0 !type !8 !type !9 {
  %2 = alloca i32, align 4
  %3 = alloca ptr, align 8
  store i32 %0, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  %5 = icmp sgt i32 %4, 0
  %6 = zext i1 %5 to i64
  %7 = select i1 %5, ptr @foo, ptr @bar
  store ptr %7, ptr %3, align 8
  %8 = load ptr, ptr %3, align 8
  %9 = call i1 @llvm.type.test(ptr %8, metadata !"_ZTSFiiE"), !nosanitize !10
  br i1 %9, label %11, label %10, !nosanitize !10

10:
  call void @llvm.ubsantrap(i8 2) #4, !nosanitize !10
  unreachable, !nosanitize !10

11:
  %12 = load i32, ptr %2, align 4
  %13 = call i32 %8(i32 noundef %12)
  ret i32 %13
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.ubsantrap(i8 immarg)

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+v8a,-fmv" }
attributes #1 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+v8a,-fmv" }
attributes #4 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 4, !"CFI Canonical Jump Tables", i32 0}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{i32 1, !"ThinLTO", i32 0}
!6 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!8 = !{i64 0, !"_ZTSFiiE"}
!9 = !{i64 0, !"_ZTSFiiE.generalized"}
!10 = !{}
