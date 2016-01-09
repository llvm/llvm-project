; RUN: opt < %s -O3 | llc -march=aarch64 -mtriple=aarch64-none-linux-gnu -stop-after branch-folder -o /dev/null | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; Function Attrs: nounwind
define void @test(float* %test_arr) #0 {
entry:
  %test_arr.addr = alloca float*, align 8
  store float* %test_arr, float** %test_arr.addr, align 8
  %0 = load float*, float** %test_arr.addr, align 8
  %arrayidx = getelementptr inbounds float, float* %0, i64 0
  %1 = load float, float* %arrayidx, align 4
  %2 = load float*, float** %test_arr.addr, align 8
  %arrayidx1 = getelementptr inbounds float, float* %2, i64 1
  %3 = load float, float* %arrayidx1, align 4
  %sub = fsub float %1, %3
  %4 = load float*, float** %test_arr.addr, align 8
  %arrayidx2 = getelementptr inbounds float, float* %4, i64 0
  store float %sub, float* %arrayidx2, align 4
  ret void
}

; Function Attrs: nounwind
define void @foo(i32 %a, i32 %b, float* %foo_arr) #0 {
; CHECK: (load 4 from %ir.arrayidx1.i2), (load 4 from %ir.arrayidx1.i)
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %foo_arr.addr = alloca float*, align 8
  store i32 %a, i32* %a.addr, align 4
  store i32 %b, i32* %b.addr, align 4
  store float* %foo_arr, float** %foo_arr.addr, align 8
  %0 = load i32, i32* %a.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %1 = load float*, float** %foo_arr.addr, align 8
  call void @test(float* %1)
  br label %if.end3

if.end:                                           ; preds = %entry
  %2 = load i32, i32* %b.addr, align 4
  %cmp1 = icmp sgt i32 %2, 0
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:                                         ; preds = %if.end
  %3 = load float*, float** %foo_arr.addr, align 8
  call void @test(float* %3)
  br label %if.end3

if.end3:                                          ; preds = %if.then, %if.then2, %if.end
  ret void
}
