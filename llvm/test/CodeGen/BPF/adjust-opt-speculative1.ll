; RUN: opt -O2 -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc -mcpu=v1 %t1 -o - | FileCheck -check-prefixes=CHECK-COMMON,CHECK %s
; RUN: opt -O2 -mtriple=bpf-pc-linux -bpf-disable-avoid-speculation %s | llvm-dis > %t1
; RUN: llc -mcpu=v1 %t1 -o - | FileCheck -check-prefixes=CHECK-COMMON,CHECK-DISABLE %s
;
; Source:
;   unsigned long foo();
;   ptr test(ptr p) {
;     unsigned long ret = foo();
;     if (ret <= 7)
;       p += ret;
;     return p;
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm -Xclang -disable-llvm-passes test.c

; Function Attrs: nounwind
define dso_local ptr @test(ptr %p) {
entry:
  %p.addr = alloca ptr, align 8
  %ret = alloca i64, align 8
  store ptr %p, ptr %p.addr, align 8, !tbaa !2
  call void @llvm.lifetime.start.p0(i64 8, ptr %ret)
  %call = call i64 @foo()
  store i64 %call, ptr %ret, align 8, !tbaa !6
  %0 = load i64, ptr %ret, align 8, !tbaa !6
  %cmp = icmp ule i64 %0, 7
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = load i64, ptr %ret, align 8, !tbaa !6
  %2 = load ptr, ptr %p.addr, align 8, !tbaa !2
  %add.ptr = getelementptr i8, ptr %2, i64 %1
  store ptr %add.ptr, ptr %p.addr, align 8, !tbaa !2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %3 = load ptr, ptr %p.addr, align 8, !tbaa !2
  call void @llvm.lifetime.end.p0(i64 8, ptr %ret)
  ret ptr %3
}
; CHECK-COMMON:  [[REG6:r[0-9]+]] = r1
; CHECK-COMMON:  call foo

; CHECK:         if r0 > 7 goto [[LABEL:.*]]
; CHECK:         [[REG6]] += r0
; CHECK:         [[LABEL]]:
; CHECK:         r0 = [[REG6]]

; CHECK-DISABLE: [[REG1:r[0-9]+]] = 8
; CHECK-DISABLE: if [[REG1]] > r0 goto [[LABEL:.*]]
; CHECK-DISABLE: r0 = 0
; CHECK-DISABLE: [[LABEL]]:
; CHECK-DISABLE: [[REG6]] += r0
; CHECK-DISABLE: r0 = [[REG6]]

; CHECK-COMMON:  exit

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

declare dso_local i64 @foo(...)

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git ca9c5433a6c31e372092fcd8bfd0e4fddd7e8784)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !4, i64 0}
