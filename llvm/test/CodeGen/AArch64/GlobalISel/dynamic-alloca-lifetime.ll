; RUN: llc -mtriple aarch64-unknown-unknown -global-isel \
; RUN:     -no-stack-coloring=false -pass-remarks-missed=gisel* < %s \
; RUN:      2>&1 | FileCheck %s

; Same as the dynamic-alloca-lifetime.ll X86 test, which was used to fix a bug
; in stack colouring + lifetime markers.

; This test crashed in PEI because the stack protector was dead.
; This was due to it being colored, which was in turn due to incorrect
; lifetimes being applied to the stack protector frame index.

; CHECK: stack_chk_guard
; CHECK-NOT: remark{{.*}}foo

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #0

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #0

; Function Attrs: ssp
define void @foo(i1 %cond1, i1 %cond2) #1 {
entry:
  %bitmapBuffer = alloca [8192 x i8], align 1
  br i1 %cond1, label %end1, label %bb1

bb1:
  %bitmapBuffer229 = alloca [8192 x i8], align 1
  br i1 %cond2, label %end1, label %if.else130

end1:
  ret void

if.else130:                                       ; preds = %bb1
  call void @llvm.lifetime.start.p0(i64 8192, ptr %bitmapBuffer) #0
  call void @llvm.lifetime.end.p0(i64 8192, ptr %bitmapBuffer) #0
  call void @llvm.lifetime.start.p0(i64 8192, ptr %bitmapBuffer229) #0
  call void @llvm.lifetime.end.p0(i64 8192, ptr %bitmapBuffer229) #0
  br label %end1
}

declare void @bar()

attributes #0 = { nounwind }
attributes #1 = { ssp }
