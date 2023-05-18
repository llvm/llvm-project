; RUN: llc < %s -mcpu=generic -mtriple=i686-- -mattr=+cmov,cx16 -verify-machineinstrs | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

define void @func(i32 %argc, ptr %argv) nounwind {
entry:
	%argc.addr = alloca i32		; <ptr> [#uses=1]
	%argv.addr = alloca ptr		; <ptr> [#uses=1]
	%val1 = alloca i32		; <ptr> [#uses=2]
	%val2 = alloca i32		; <ptr> [#uses=15]
	%andt = alloca i32		; <ptr> [#uses=2]
	%ort = alloca i32		; <ptr> [#uses=2]
	%xort = alloca i32		; <ptr> [#uses=2]
	%old = alloca i32		; <ptr> [#uses=18]
	%temp = alloca i32		; <ptr> [#uses=2]
	%temp64 = alloca i64
	store i32 %argc, ptr %argc.addr
	store ptr %argv, ptr %argv.addr
	store i32 0, ptr %val1
	store i32 31, ptr %val2
	store i32 3855, ptr %andt
	store i32 3855, ptr %ort
	store i32 3855, ptr %xort
	store i32 4, ptr %temp
	%tmp = load i32, ptr %temp
        ; CHECK: lock
        ; CHECK: xaddl
  %0 = atomicrmw add ptr %val1, i32 %tmp monotonic
	store i32 %0, ptr %old
        ; CHECK: lock
        ; CHECK: xaddl
  %1 = atomicrmw sub ptr %val2, i32 30 monotonic
	store i32 %1, ptr %old
        ; CHECK: lock
        ; CHECK: xaddl
  %2 = atomicrmw add ptr %val2, i32 1 monotonic
	store i32 %2, ptr %old
        ; CHECK: lock
        ; CHECK: xaddl
  %3 = atomicrmw sub ptr %val2, i32 1 monotonic
	store i32 %3, ptr %old
        ; CHECK: andl
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %4 = atomicrmw and ptr %andt, i32 4080 monotonic
	store i32 %4, ptr %old
        ; CHECK: orl
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %5 = atomicrmw or ptr %ort, i32 4080 monotonic
	store i32 %5, ptr %old
        ; CHECK: xorl
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %6 = atomicrmw xor ptr %xort, i32 4080 monotonic
	store i32 %6, ptr %old
        ; CHECK: cmov
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %7 = atomicrmw min ptr %val2, i32 16 monotonic
	store i32 %7, ptr %old
	%neg = sub i32 0, 1		; <i32> [#uses=1]
        ; CHECK: cmov
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %8 = atomicrmw min ptr %val2, i32 %neg monotonic
	store i32 %8, ptr %old
        ; CHECK: cmov
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %9 = atomicrmw max ptr %val2, i32 1 monotonic
	store i32 %9, ptr %old
        ; CHECK: cmov
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %10 = atomicrmw max ptr %val2, i32 0 monotonic
	store i32 %10, ptr %old
        ; CHECK: cmov
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %11 = atomicrmw umax ptr %val2, i32 65535 monotonic
	store i32 %11, ptr %old
        ; CHECK: cmov
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %12 = atomicrmw umax ptr %val2, i32 10 monotonic
	store i32 %12, ptr %old
        ; CHECK: cmov
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %13 = atomicrmw umin ptr %val2, i32 1 monotonic
	store i32 %13, ptr %old
        ; CHECK: cmov
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %14 = atomicrmw umin ptr %val2, i32 10 monotonic
	store i32 %14, ptr %old
        ; CHECK: xchgl   %{{.*}}, {{.*}}(%esp)
  %15 = atomicrmw xchg ptr %val2, i32 1976 monotonic
	store i32 %15, ptr %old
	%neg1 = sub i32 0, 10		; <i32> [#uses=1]
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %pair16 = cmpxchg ptr %val2, i32 %neg1, i32 1 monotonic monotonic
  %16 = extractvalue { i32, i1 } %pair16, 0
	store i32 %16, ptr %old
        ; CHECK: lock
        ; CHECK: cmpxchgl
  %pair17 = cmpxchg ptr %val2, i32 1976, i32 1 monotonic monotonic
  %17 = extractvalue { i32, i1 } %pair17, 0
	store i32 %17, ptr %old
        ; CHECK: movl  [[R17atomic:.*]], %eax
        ; CHECK: movl %eax, %[[R17mask:[a-z]*]]
        ; CHECK: notl %[[R17mask]]
        ; CHECK: orl $-1402, %[[R17mask]]
        ; CHECK: lock
        ; CHECK: cmpxchgl	%[[R17mask]], [[R17atomic]]
        ; CHECK: jne
        ; CHECK: movl	%eax,
  %18 = atomicrmw nand ptr %val2, i32 1401 monotonic
  store i32 %18, ptr %old
        ; CHECK: notl
        ; CHECK: notl
        ; CHECK: orl $252645135
        ; CHECK: orl $252645135
        ; CHECK: lock
        ; CHECK: cmpxchg8b
  %19 = atomicrmw nand ptr %temp64, i64 17361641481138401520 monotonic
  store i64 %19, ptr %temp64
	ret void
}

define void @test2(ptr addrspace(256) nocapture %P) nounwind {
entry:
; CHECK: lock
; CHECK:	cmpxchgl	%{{.*}}, %gs:(%{{.*}})

  %pair0 = cmpxchg ptr addrspace(256) %P, i32 0, i32 1 monotonic monotonic
  %0 = extractvalue { i32, i1 } %pair0, 0
  ret void
}
