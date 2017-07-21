; Function Attrs: nounwind
; RUN: llc < %s -march=r600 -mcpu=redwood  | FileCheck %s
;
; CFG flattening should use parallel-or to generate branch conditions and
; then merge if-regions with the same bodies.

; FIXME: For some reason having the allocas here allowed the flatten cfg pass
; to do its transformation, however now that we are using local memory for
; allocas, the transformation isn't happening.
; XFAIL: *
;
; CHECK: OR_INT
; CHECK-NEXT: OR_INT
; CHECK-NEXT: OR_INT
define amdgpu_kernel void @_Z9chk1D_512v() #0 {
entry:
  %a0 = alloca i32, align 4
  %b0 = alloca i32, align 4
  %c0 = alloca i32, align 4
  %d0 = alloca i32, align 4
  %a1 = alloca i32, align 4
  %b1 = alloca i32, align 4
  %c1 = alloca i32, align 4
  %d1 = alloca i32, align 4
  %data = alloca i32, align 4
  %0 = load i32, i32* %a0, align 4
  %1 = load i32, i32* %b0, align 4
  %cmp = icmp ne i32 %0, %1
  br i1 %cmp, label %land.lhs.true, label %if.else

land.lhs.true:                                    ; preds = %entry
  %2 = load i32, i32* %c0, align 4
  %3 = load i32, i32* %d0, align 4
  %cmp1 = icmp ne i32 %2, %3
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %land.lhs.true
  br label %if.end

if.else:                                          ; preds = %land.lhs.true, %entry
  store i32 1, i32* %data, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %4 = load i32, i32* %a1, align 4
  %5 = load i32, i32* %b1, align 4
  %cmp2 = icmp ne i32 %4, %5
  br i1 %cmp2, label %land.lhs.true3, label %if.else6

land.lhs.true3:                                   ; preds = %if.end
  %6 = load i32, i32* %c1, align 4
  %7 = load i32, i32* %d1, align 4
  %cmp4 = icmp ne i32 %6, %7
  br i1 %cmp4, label %if.then5, label %if.else6

if.then5:                                         ; preds = %land.lhs.true3
  br label %if.end7

if.else6:                                         ; preds = %land.lhs.true3, %if.end
  store i32 1, i32* %data, align 4
  br label %if.end7

if.end7:                                          ; preds = %if.else6, %if.then5
  ret void
}

