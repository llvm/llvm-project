; RUN: llc < %s -march=x86-64 -mcpu=core-avx-i -mattr=+rdrand | FileCheck %s
declare {i16, i32} @llvm.x86.rdrand.16()
declare {i32, i32} @llvm.x86.rdrand.32()
declare {i64, i32} @llvm.x86.rdrand.64()

define i32 @_rdrand16_step(i16* %random_val) {
  %call = call {i16, i32} @llvm.x86.rdrand.16()
  %randval = extractvalue {i16, i32} %call, 0
  store i16 %randval, i16* %random_val
  %isvalid = extractvalue {i16, i32} %call, 1
  ret i32 %isvalid
; CHECK: _rdrand16_step:
; CHECK: rdrandw	%ax
; CHECK: movw	%ax, (%r[[A0:di|cx]])
; CHECK: movzwl	%ax, %ecx
; CHECK: movl	$1, %eax
; CHECK: cmovael	%ecx, %eax
; CHECK: ret
}

define i32 @_rdrand32_step(i32* %random_val) {
  %call = call {i32, i32} @llvm.x86.rdrand.32()
  %randval = extractvalue {i32, i32} %call, 0
  store i32 %randval, i32* %random_val
  %isvalid = extractvalue {i32, i32} %call, 1
  ret i32 %isvalid
; CHECK: _rdrand32_step:
; CHECK: rdrandl	%e[[T0:[a-z]+]]
; CHECK: movl	%e[[T0]], (%r[[A0]])
; CHECK: movl	$1, %eax
; CHECK: cmovael	%e[[T0]], %eax
; CHECK: ret
}

define i32 @_rdrand64_step(i64* %random_val) {
  %call = call {i64, i32} @llvm.x86.rdrand.64()
  %randval = extractvalue {i64, i32} %call, 0
  store i64 %randval, i64* %random_val
  %isvalid = extractvalue {i64, i32} %call, 1
  ret i32 %isvalid
; CHECK: _rdrand64_step:
; CHECK: rdrandq	%r[[T1:[[a-z]+]]
; CHECK: movq	%r[[T1]], (%r[[A0]])
; CHECK: movl	$1, %eax
; CHECK: cmovael	%e[[T1]], %eax
; CHECK: ret
}

; Check that MachineCSE doesn't eliminate duplicate rdrand instructions.
define i32 @CSE() nounwind {
 %rand1 = tail call { i32, i32 } @llvm.x86.rdrand.32() nounwind
 %v1 = extractvalue { i32, i32 } %rand1, 0
 %rand2 = tail call { i32, i32 } @llvm.x86.rdrand.32() nounwind
 %v2 = extractvalue { i32, i32 } %rand2, 0
 %add = add i32 %v2, %v1
 ret i32 %add
; CHECK: CSE:
; CHECK: rdrandl
; CHECK: rdrandl
}

; Check that MachineLICM doesn't hoist rdrand instructions.
define void @loop(i32* %p, i32 %n) nounwind {
entry:
  %tobool1 = icmp eq i32 %n, 0
  br i1 %tobool1, label %while.end, label %while.body

while.body:                                       ; preds = %entry, %while.body
  %p.addr.03 = phi i32* [ %incdec.ptr, %while.body ], [ %p, %entry ]
  %n.addr.02 = phi i32 [ %dec, %while.body ], [ %n, %entry ]
  %dec = add nsw i32 %n.addr.02, -1
  %incdec.ptr = getelementptr inbounds i32* %p.addr.03, i64 1
  %rand = tail call { i32, i32 } @llvm.x86.rdrand.32() nounwind
  %v1 = extractvalue { i32, i32 } %rand, 0
  store i32 %v1, i32* %p.addr.03, align 4
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  ret void
; CHECK: loop:
; CHECK-NOT: rdrandl
; CHECK: This Inner Loop Header: Depth=1
; CHECK: rdrandl
}
