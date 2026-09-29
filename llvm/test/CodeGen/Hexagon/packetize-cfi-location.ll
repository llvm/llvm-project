; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target triple = "hexagon"
%type.0 = type { i32, ptr, i32, i32, i32 }

; Check that CFI is before the packet with call+allocframe.
; CHECK-LABEL: danny:
; CHECK: cfi_def_cfa
; CHECK: call throw
; CHECK-NEXT: allocframe

; Expect packet:
; {
;   call throw
;   allocframe(#0)
; }

define ptr @danny(ptr %p0, i32 %p1) #0 {
entry:
  %t0 = getelementptr inbounds %type.0, ptr %p0, i32 0, i32 4
  %t1 = load i32, ptr %t0, align 4
  %th = icmp ugt i32 %t1, %p1
  br i1 %th, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void @throw(ptr nonnull %p0)
  unreachable

if.end:                                           ; preds = %entry
  %t6 = getelementptr inbounds %type.0, ptr %p0, i32 0, i32 3
  %t2 = load i32, ptr %t6, align 4
  %t9 = add i32 %t2, %p1
  %ta = lshr i32 %t9, 4
  %tb = and i32 %t9, 15
  %t7 = getelementptr inbounds %type.0, ptr %p0, i32 0, i32 2
  %t3 = load i32, ptr %t7, align 4
  %tc = icmp ult i32 %ta, %t3
  %td = select i1 %tc, i32 0, i32 %t3
  %te = sub i32 %ta, %td
  %t8 = getelementptr inbounds %type.0, ptr %p0, i32 0, i32 1
  %t4 = load ptr, ptr %t8, align 4
  %tf = getelementptr inbounds ptr, ptr %t4, i32 %te
  %t5 = load ptr, ptr %tf, align 4
  %tg = getelementptr inbounds i8, ptr %t5, i32 %tb
  ret ptr %tg
}

; Check that CFI is after allocframe.
; CHECK-LABEL: sammy:
; CHECK: allocframe
; CHECK: cfi_def_cfa

define void @sammy(ptr %p0, i32 %p1) #0 {
entry:
  %t0 = icmp sgt i32 %p1, 0
  br i1 %t0, label %if.then, label %if.else
if.then:
  call void @throw(ptr nonnull %p0)
  br label %if.end
if.else:
  call void @nothrow() #2
  br label %if.end
if.end:
  ret void
}

declare void @throw(ptr) #1
declare void @nothrow() #2

attributes #0 = { "target-cpu"="hexagonv55" }
attributes #1 = { noreturn "target-cpu"="hexagonv55" }
attributes #2 = { nounwind "target-cpu"="hexagonv55" }
