; RUN: opt -O2 -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc %t1 -o - | FileCheck -check-prefixes=CHECK %s
; RUN: opt -passes='default<O2>' -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc %t1 -o - | FileCheck -check-prefixes=CHECK %s
; RUN: opt -O2 -mtriple=bpf-pc-linux -bpf-disable-serialize-icmp %s | llvm-dis > %t1
; RUN: llc %t1 -o - | FileCheck -check-prefixes=CHECK-DISABLE %s
; RUN: opt -passes='default<O2>' -mtriple=bpf-pc-linux -bpf-disable-serialize-icmp %s | llvm-dis > %t1
; RUN: llc %t1 -o - | FileCheck -check-prefixes=CHECK-DISABLE %s
;
; Source:
;   int foo();
;   int bar(int);
;   int test() {
;     int ret = foo();
;     if (ret <= 0 || ret > 7)
;       return 0;
;     return bar(ret);
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm -Xclang -disable-llvm-passes test.c

; Function Attrs: nounwind
define dso_local i32 @test() #0 {
entry:
  %retval = alloca i32, align 4
  %ret = alloca i32, align 4
  %cleanup.dest.slot = alloca i32, align 4
  %0 = bitcast i32* %ret to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  %call = call i32 bitcast (i32 (...)* @foo to i32 ()*)()
  store i32 %call, i32* %ret, align 4, !tbaa !2
  %1 = load i32, i32* %ret, align 4, !tbaa !2
  %cmp = icmp sle i32 %1, 0
  br i1 %cmp, label %if.then, label %lor.lhs.false

; CHECK:         [[REG1:r[0-9]+]] <<= 32
; CHECK:         [[REG1]] s>>= 32
; CHECK:         [[REG2:r[0-9]+]] = 1
; CHECK:         if [[REG2]] s> [[REG1]] goto
; CHECK:         if [[REG1]] s> 7 goto

; CHECK-DISABLE: [[REG1:r[0-9]+]] += -1
; CHECK-DISABLE: [[REG1]] <<= 32
; CHECK-DISABLE: [[REG1]] >>= 32
; CHECK-DISABLE: if [[REG1]] > 6 goto

lor.lhs.false:                                    ; preds = %entry
  %2 = load i32, i32* %ret, align 4, !tbaa !2
  %cmp1 = icmp sgt i32 %2, 7
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %lor.lhs.false, %entry
  store i32 0, i32* %retval, align 4
  store i32 1, i32* %cleanup.dest.slot, align 4
  br label %cleanup

if.end:                                           ; preds = %lor.lhs.false
  %3 = load i32, i32* %ret, align 4, !tbaa !2
  %call2 = call i32 @bar(i32 %3)
  store i32 %call2, i32* %retval, align 4
  store i32 1, i32* %cleanup.dest.slot, align 4
  br label %cleanup

cleanup:                                          ; preds = %if.end, %if.then
  %4 = bitcast i32* %ret to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %4) #3
  %5 = load i32, i32* %retval, align 4
  ret i32 %5
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local i32 @foo(...) #2

declare dso_local i32 @bar(i32) #2

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 12.0.0 (https://github.com/llvm/llvm-project.git ca9c5433a6c31e372092fcd8bfd0e4fddd7e8784)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
