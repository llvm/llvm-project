; RUN: opt -O2 -S -mtriple=bpf-pc-linux %s -o %t1
; RUN: llc %t1 -o - | FileCheck -check-prefixes=CHECK,CHECK-V1 %s
; RUN: opt -O2 -S -mtriple=bpf-pc-linux %s -o %t1
; RUN: llc %t1 -mcpu=v3 -o - | FileCheck -check-prefixes=CHECK,CHECK-V3 %s
;
; Source:
;   unsigned bar(unsigned);
;   unsigned int test(unsigned *p) {
;     if (*p <= 1 || *p >= 7)
;       return 0;
;     return bar(*p);
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm -Xclang -disable-llvm-passes test.c

; Function Attrs: nounwind
define dso_local i32 @test(i32* noundef %p) #0 {
entry:
  %retval = alloca i32, align 4
  %p.addr = alloca i32*, align 8
  store i32* %p, i32** %p.addr, align 8, !tbaa !3
  %0 = load i32*, i32** %p.addr, align 8, !tbaa !3
  %1 = load i32, i32* %0, align 4, !tbaa !7
  %cmp = icmp ule i32 %1, 1
  br i1 %cmp, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %2 = load i32*, i32** %p.addr, align 8, !tbaa !3
  %3 = load i32, i32* %2, align 4, !tbaa !7
  %cmp1 = icmp uge i32 %3, 7
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %lor.lhs.false, %entry
  store i32 0, i32* %retval, align 4
  br label %return

if.end:                                           ; preds = %lor.lhs.false
  %4 = load i32*, i32** %p.addr, align 8, !tbaa !3
  %5 = load i32, i32* %4, align 4, !tbaa !7
  %call = call i32 @bar(i32 noundef %5)
  store i32 %call, i32* %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %6 = load i32, i32* %retval, align 4
  ret i32 %6
}

; CHECK-LABEL: test
; CHECK-V1:    if r[[#]] > r[[#]] goto
; CHECK-V1:    if r[[#]] > 6 goto
; CHECK-V3:    if w[[#]] < 2 goto
; CHECK-V3:    if w[[#]] > 6 goto

declare dso_local i32 @bar(i32 noundef) #1

attributes #0 = { nounwind "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 15.0.0 (https://github.com/llvm/llvm-project.git 2a25e1af85f3138f70888c4c3f359c6a09e3cfe5)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"any pointer", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !5, i64 0}
