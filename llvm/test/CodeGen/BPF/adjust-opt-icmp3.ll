; RUN: opt -O2 -S -mtriple=bpf-pc-linux %s -o %t1
; RUN: llc %t1 -o - | FileCheck -check-prefixes=CHECK,CHECK-V1 %s
; RUN: opt -O2 -S -mtriple=bpf-pc-linux %s -o %t1
; RUN: llc %t1 -mcpu=v3 -o - | FileCheck -check-prefixes=CHECK,CHECK-V3 %s
;
; Source:
;   int test1(unsigned long a) {
;     if ((unsigned)a <= 3) return 2;
;     return 3;
;   }
;   int test2(unsigned long a) {
;     if ((unsigned)a < 4) return 2;
;     return 3;
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm -Xclang -disable-llvm-passes test.c

; Function Attrs: nounwind
define dso_local i32 @test1(i64 %a) #0 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i64, align 8
  store i64 %a, i64* %a.addr, align 8, !tbaa !3
  %0 = load i64, i64* %a.addr, align 8, !tbaa !3
  %conv = trunc i64 %0 to i32
  %cmp = icmp ule i32 %conv, 3
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 2, i32* %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  store i32 3, i32* %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %1 = load i32, i32* %retval, align 4
  ret i32 %1
}

; CHECK-LABEL: test1
; CHECK-V1:    if r[[#]] > r[[#]] goto
; CHECK-V3:    if w[[#]] < 4 goto

; Function Attrs: nounwind
define dso_local i32 @test2(i64 %a) #0 {
entry:
  %retval = alloca i32, align 4
  %a.addr = alloca i64, align 8
  store i64 %a, i64* %a.addr, align 8, !tbaa !3
  %0 = load i64, i64* %a.addr, align 8, !tbaa !3
  %conv = trunc i64 %0 to i32
  %cmp = icmp ult i32 %conv, 4
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 2, i32* %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  store i32 3, i32* %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %1 = load i32, i32* %retval, align 4
  ret i32 %1
}

; CHECK-LABEL: test2
; CHECK-V1:    if r[[#]] > r[[#]] goto
; CHECK-V3:    if w[[#]] < 4 goto

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git b7892f95881c891032742e0cd81861b845512653)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"long", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
