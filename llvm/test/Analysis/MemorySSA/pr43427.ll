; RUN: opt -disable-output -licm -print-memoryssa -enable-mssa-loop-dependency=true < %s 2>&1 | FileCheck %s

; CHECK-LABEL: @f()
; CHECK: 8 = MemoryPhi(
; CHECK: 7 = MemoryPhi(
; CHECK: 9 = MemoryPhi(
define void @f() {
entry:
  %e = alloca i16, align 1
  br label %lbl1

lbl1:                                             ; preds = %if.else, %cleanup, %entry
  store i16 undef, i16* %e, align 1
  call void @g()
  br i1 undef, label %for.end, label %if.else

for.end:                                          ; preds = %lbl1
  br i1 undef, label %lbl3, label %lbl2

lbl2:                                             ; preds = %lbl3, %for.end
  br label %lbl3

lbl3:                                             ; preds = %lbl2, %for.end
  br i1 undef, label %lbl2, label %cleanup

cleanup:                                          ; preds = %lbl3
  %cleanup.dest = load i32, i32* undef, align 1
  %switch = icmp ult i32 %cleanup.dest, 1
  br i1 %switch, label %cleanup.cont, label %lbl1

cleanup.cont:                                     ; preds = %cleanup
  call void @llvm.lifetime.end.p0i8(i64 1, i8* null)
  ret void

if.else:                                          ; preds = %lbl1
  br label %lbl1
}

declare void @g()

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
