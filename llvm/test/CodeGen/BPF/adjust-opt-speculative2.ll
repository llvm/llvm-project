; RUN: opt -O2 -mtriple=bpf-pc-linux %s | llvm-dis > %t1
; RUN: llc %t1 -o - | FileCheck -check-prefixes=CHECK-COMMON,CHECK %s
; RUN: opt -O2 -mtriple=bpf-pc-linux -bpf-disable-avoid-speculation %s | llvm-dis > %t1
; RUN: llc %t1 -o - | FileCheck -check-prefixes=CHECK-COMMON,CHECK-DISABLE %s
;
; Source:
;   unsigned foo();
;   void *test(void *p) {
;     unsigned ret = foo();
;     if (ret <= 7)
;       p += ret;
;     return p;
;   }
; Compilation flag:
;   clang -target bpf -O2 -S -emit-llvm -Xclang -disable-llvm-passes test.c

; Function Attrs: nounwind
define dso_local i8* @test(i8* %p) #0 {
entry:
  %p.addr = alloca i8*, align 8
  %ret = alloca i32, align 4
  store i8* %p, i8** %p.addr, align 8, !tbaa !2
  %0 = bitcast i32* %ret to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0) #3
  %call = call i32 bitcast (i32 (...)* @foo to i32 ()*)()
  store i32 %call, i32* %ret, align 4, !tbaa !6
  %1 = load i32, i32* %ret, align 4, !tbaa !6
  %cmp = icmp ule i32 %1, 7
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %2 = load i32, i32* %ret, align 4, !tbaa !6
  %3 = load i8*, i8** %p.addr, align 8, !tbaa !2
  %idx.ext = zext i32 %2 to i64
  %add.ptr = getelementptr i8, i8* %3, i64 %idx.ext
  store i8* %add.ptr, i8** %p.addr, align 8, !tbaa !2
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %4 = load i8*, i8** %p.addr, align 8, !tbaa !2
  %5 = bitcast i32* %ret to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %5) #3
  ret i8* %4
}

; CHECK-COMMON:  [[REG6:r[0-9]+]] = r1
; CHECK-COMMON:  call foo

; CHECK:         r0 <<= 32
; CHECK:         r0 >>= 32
; CHECK:         if r0 > 7 goto [[LABEL:.*]]
; CHECK:         [[REG6]] += r0
; CHECK:         [[LABEL]]:
; CHECK:         r0 = [[REG6]]

; CHECK-DISABLE: [[REG1:r[0-9]+]] = r0
; CHECK-DISABLE: [[REG1]] <<= 32
; CHECK-DISABLE: [[REG1]] >>= 32
; CHECK-DISABLE: r0 = [[REG6]]
; CHECK-DISABLE: r0 += [[REG1]]
; CHECK-DISABLE: [[REG2:r[0-9]+]] = 8
; CHECK-DISABLE: if [[REG2]] > [[REG1]] goto [[LABEL:.*]]
; CHECK-DISABLE: r0 = [[REG6]]
; CHECK-DISABLE: [[LABEL]]:

; CHECK-COMMON:  exit

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local i32 @foo(...) #2

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
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}
