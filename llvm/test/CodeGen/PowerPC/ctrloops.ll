target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-freebsd10.0"
; RUN: llc < %s -march=ppc64 -relocation-model=pic | FileCheck %s

@a = common global i32 0, align 4

define void @test1(i32 %c) nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load volatile i32, i32* @a, align 4
  %add = add nsw i32 %0, %c
  store volatile i32 %add, i32* @a, align 4
  %inc = add nsw i32 %i.01, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
; CHECK: @test1
; CHECK-NOT: or 3, 3, 3
; CHECK: mtctr
; CHECK-NOT: addi {[0-9]+}
; CHECK-NOT: cmplwi
; CHECK: bdnz
}

define void @test2(i32 %c, i32 %d) nounwind {
entry:
  %cmp1 = icmp sgt i32 %d, 0
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %0 = load volatile i32, i32* @a, align 4
  %add = add nsw i32 %0, %c
  store volatile i32 %add, i32* @a, align 4
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, %d
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test2
; CHECK: mtctr
; CHECK-NOT: addi {[0-9]+}
; CHECK-NOT: cmplwi
; CHECK: bdnz
}

define void @test3(i32 %c, i32 %d) nounwind {
entry:
  %cmp1 = icmp sgt i32 %d, 0
  br i1 %cmp1, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %mul = mul nsw i32 %i.02, %c
  %0 = load volatile i32, i32* @a, align 4
  %add = add nsw i32 %0, %mul
  store volatile i32 %add, i32* @a, align 4
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, %d
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test3
; CHECK: mtctr
; CHECK-NOT: addi {[0-9]+}
; CHECK-NOT: cmplwi
; CHECK: bdnz
}

@tls_var = external thread_local global i8

define i32 @test4(i32 %inp) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %phi = phi i32 [ %dec, %for.body ], [ %inp, %entry ]
  %load = ptrtoint i8* @tls_var to i32
  %val = add i32 %load, %phi
  %dec = add i32 %phi, -1
  %cmp = icmp sgt i32 %phi, 1
  br i1 %cmp, label %for.body, label %return

return:                                           ; preds = %for.body
  ret i32 %val
; CHECK-LABEL: @test4
; CHECK: mtctr
; CHECK: bdnz
; CHECK: __tls_get_addr
}
