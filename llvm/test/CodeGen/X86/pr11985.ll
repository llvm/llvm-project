; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=prescott | FileCheck %s

define float @foo(i8* nocapture %buf, float %a, float %b) nounwind uwtable {
entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %buf, i8* blockaddress(@foo, %out), i64 22, i32 1, i1 false)
  br label %out

out:                                              ; preds = %entry
  %add = fadd float %a, %b
  ret float %add
; CHECK: foo
; CHECK: movw .L{{.*}}+20(%rip), %{{.*}}
; CHECK: movl .L{{.*}}+16(%rip), %{{.*}}
; CHECK: movq .L{{.*}}+8(%rip), %{{.*}}
; CHECK: movq .L{{.*}}(%rip), %{{.*}}
; CHECK: ret
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i32, i1) nounwind
