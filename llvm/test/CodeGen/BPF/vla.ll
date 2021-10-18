; RUN: opt --bpf-ir-peephole -mtriple=bpf-pc-linux -S %s | FileCheck %s
; Source:
;   #define AA 40
;   struct t {
;     char a[20];
;   };
;   void foo(void *);
;
;   int test1() {
;     const int a = 8;
;     char tmp[AA + sizeof(struct t) + a];
;     foo(tmp);
;     return 0;
;   }
;
;   int test2(int b) {
;     const int a = 8;
;     char tmp[a + b];
;     foo(tmp);
;     return 0;
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm t.c -Xclang -disable-llvm-passes

source_filename = "t.c"
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "bpf"

; Function Attrs: nounwind
define dso_local i32 @test1() #0 {
entry:
  %a = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %0 = bitcast i32* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4
  store i32 8, i32* %a, align 4, !tbaa !3
  %1 = call i8* @llvm.stacksave()
  store i8* %1, i8** %saved_stack, align 8
  %vla = alloca i8, i64 68, align 1
  call void @foo(i8* %vla)
  %2 = load i8*, i8** %saved_stack, align 8
  call void @llvm.stackrestore(i8* %2)
  %3 = bitcast i32* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %3) #4
  ret i32 0
}

; CHECK:       define dso_local i32 @test1
; CHECK-NOT:   %[[#]] = call i8* @llvm.stacksave()
; CHECK-NOT:   store i8* %[[#]], i8** %saved_stack, align 8
; CHECK-NOT:   %[[#]] = load i8*, i8** %saved_stack, align 8
; CHECK-NOT:   call void @llvm.stackrestore(i8* %[[#]])

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nofree nosync nounwind willreturn
declare i8* @llvm.stacksave() #2

declare dso_local void @foo(i8*) #3

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.stackrestore(i8*) #2

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind
define dso_local i32 @test2(i32 %b) #0 {
entry:
  %b.addr = alloca i32, align 4
  %a = alloca i32, align 4
  %saved_stack = alloca i8*, align 8
  %__vla_expr0 = alloca i64, align 8
  store i32 %b, i32* %b.addr, align 4, !tbaa !3
  %0 = bitcast i32* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #4
  store i32 8, i32* %a, align 4, !tbaa !3
  %1 = load i32, i32* %b.addr, align 4, !tbaa !3
  %add = add nsw i32 8, %1
  %2 = zext i32 %add to i64
  %3 = call i8* @llvm.stacksave()
  store i8* %3, i8** %saved_stack, align 8
  %vla = alloca i8, i64 %2, align 1
  store i64 %2, i64* %__vla_expr0, align 8
  call void @foo(i8* %vla)
  %4 = load i8*, i8** %saved_stack, align 8
  call void @llvm.stackrestore(i8* %4)
  %5 = bitcast i32* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %5) #4
  ret i32 0
}

; CHECK:       define dso_local i32 @test2
; CHECK-NOT:   %[[#]] = call i8* @llvm.stacksave()
; CHECK-NOT:   store i8* %[[#]], i8** %saved_stack, align 8
; CHECK-NOT:   %[[#]] = load i8*, i8** %saved_stack, align 8
; CHECK-NOT:   call void @llvm.stackrestore(i8* %[[#]])

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { argmemonly nofree nosync nounwind willreturn }
attributes #2 = { nofree nosync nounwind willreturn }
attributes #3 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git 64c5d5c671fb5b5f25c464652a4eec2cf743af0d)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
