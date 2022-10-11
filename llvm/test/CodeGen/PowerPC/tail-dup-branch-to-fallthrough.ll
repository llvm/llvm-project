; RUN: llc -O2 %s -o - | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #0

declare void @f1()
declare void @f2()
declare void @f3()
declare void @f4()

; Function Attrs: nounwind
; CHECK-LABEL: tail_dup_fallthrough_with_branch
; CHECK: # %bb.{{[0-9]+}}: # %entry
; CHECK-NOT: # %{{[-_a-zA-Z0-9]+}}
; CHECK: # %bb.{{[0-9]+}}: # %entry
; CHECK-NOT: # %{{[-_a-zA-Z0-9]+}}
; CHECK: # %bb.{{[0-9]+}}: # %sw.0
; CHECK-NOT: # %{{[-_a-zA-Z0-9]+}}
; CHECK: # %sw.1
; CHECK-NOT: # %{{[-_a-zA-Z0-9]+}}
; CHECK: # %sw.default
; CHECK-NOT: # %{{[-_a-zA-Z0-9]+}}
; CHECK: # %bb.{{[0-9]+}}: # %if.then
; CHECK-NOT: # %{{[-_a-zA-Z0-9]+}}
; CHECK: # %if.else
; CHECK-NOT: # %{{[-_a-zA-Z0-9]+}}
; CHECK: .Lfunc_end0
define fastcc void @tail_dup_fallthrough_with_branch(i32 %a, i1 %b) unnamed_addr #0 {
entry:
  switch i32 %a, label %sw.default [
    i32 0, label %sw.0
    i32 1, label %sw.1
  ]

sw.0:                                         ; preds = %entry
  call void @f1() #0
  br label %dup1

sw.1:                                         ; preds = %entry
  call void @f2() #0
  br label %dup1

sw.default:                                   ; preds = %entry
  br i1 %b, label %if.then, label %if.else

if.then:                                      ; preds = %sw.default
  call void @f3() #0
  br label %dup2

if.else:                                      ; preds = %sw.default
  call void @f4() #0
  br label %dup2

dup1:                                         ; preds = %sw.0, %sw.1
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull undef) #0
  unreachable

dup2:                                         ; preds = %if.then, %if.else
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull undef) #0
  unreachable
}

attributes #0 = { nounwind }
