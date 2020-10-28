; RUN: llc -mtriple=x86_64-pc-linux-gnu -o - < %s | FileCheck --check-prefix=CHECK-TLS-FS-40 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -stack-protector-guard=tls -o - < %s | FileCheck --check-prefix=CHECK-TLS-FS-40 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -stack-protector-guard=global -o - < %s | FileCheck --check-prefix=CHECK-GLOBAL %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -stack-protector-guard-reg=fs -o - < %s | FileCheck --check-prefix=CHECK-TLS-FS-40 %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -stack-protector-guard-reg=gs -o - < %s | FileCheck --check-prefix=CHECK-GS %s
; RUN: llc -mtriple=x86_64-pc-linux-gnu -stack-protector-guard-offset=20 -o - < %s | FileCheck --check-prefix=CHECK-OFFSET %s

; CHECK-TLS-FS-40:       movq    %fs:40, %rax
; CHECK-TLS-FS-40:       movq    %fs:40, %rax
; CHECK-TLS-FS-40-NEXT:  cmpq    16(%rsp), %rax
; CHECK-TLS-FS-40-NEXT:  jne     .LBB0_2
; CHECK-TLS-FS-40:       .LBB0_2:
; CHECK-TLS-FS-40-NEXT:  .cfi_def_cfa_offset 32
; CHECK-TLS-FS-40-NEXT:  callq   __stack_chk_fail

; CHECK-GS:       movq    %gs:40, %rax
; CHECK-GS:       movq    %gs:40, %rax
; CHECK-GS-NEXT:  cmpq    16(%rsp), %rax
; CHECK-GS-NEXT:  jne     .LBB0_2
; CHECK-GS:       .LBB0_2:
; CHECK-GS-NEXT:  .cfi_def_cfa_offset 32
; CHECK-GS-NEXT:  callq   __stack_chk_fail

; CHECK-OFFSET:       movq    %fs:20, %rax
; CHECK-OFFSET:       movq    %fs:20, %rax
; CHECK-OFFSET-NEXT:  cmpq    16(%rsp), %rax
; CHECK-OFFSET-NEXT:  jne     .LBB0_2
; CHECK-OFFSET:       .LBB0_2:
; CHECK-OFFSET-NEXT:  .cfi_def_cfa_offset 32
; CHECK-OFFSET-NEXT:  callq   __stack_chk_fail

; CHECK-GLOBAL:       movq    __stack_chk_guard(%rip), %rax
; CHECK-GLOBAL:       movq    __stack_chk_guard(%rip), %rax
; CHECK-GLOBAL-NEXT:  cmpq    16(%rsp), %rax
; CHECK-GLOBAL-NEXT:  jne     .LBB0_2
; CHECK-GLOBAL:       .LBB0_2:
; CHECK-GLOBAL-NEXT:  .cfi_def_cfa_offset 32
; CHECK-GLOBAL-NEXT:  callq   __stack_chk_fail

; ModuleID = 't.c'
@.str = private unnamed_addr constant [14 x i8] c"stackoverflow\00", align 1
@a = dso_local local_unnamed_addr global i8* null, align 8

; Function Attrs: nounwind sspreq uwtable writeonly
define dso_local i32 @main() local_unnamed_addr #0 {
entry:
  %array = alloca [5 x i8], align 1
  %0 = getelementptr inbounds [5 x i8], [5 x i8]* %array, i64 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 5, i8* nonnull %0) #2
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* nonnull align 1 dereferenceable(14) %0, i8* nonnull align 1 dereferenceable(14) getelementptr inbounds ([14 x i8], [14 x i8]* @.str, i64 0, i64 0), i64 14, i1 false) #2
  store i8* %0, i8** @a, align 8
  call void @llvm.lifetime.end.p0i8(i64 5, i8* nonnull %0) #2
  ret i32 0
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

attributes #0 = { nounwind sspreq uwtable writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind }
