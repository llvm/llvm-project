; RUN: llc     < %s | FileCheck %s
; RUN: llc -O3 < %s | FileCheck %s
; RUN: llc -O3 -debug-only=stackmaps < %s 2>&1 | FileCheck -check-prefix=STACKMAPS %s
; REQUIRES: asserts

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"


; STACKMAPS: Stack Maps: callsite 2882400015
; STACKMAPS-NEXT: Stack Maps:   has 4 locations
; STACKMAPS-NEXT: Stack Maps: 		Loc 0: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 1: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 2: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 		Loc 3: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 	has 0 live-out registers
; STACKMAPS-NEXT: Stack Maps: callsite 4242
; STACKMAPS-NEXT: Stack Maps:   has 4 locations
; STACKMAPS-NEXT: Stack Maps: 		Loc 0: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 1: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 2: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 		Loc 3: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 	has 0 live-out registers
; STACKMAPS-NEXT: Stack Maps: callsite 4243
; STACKMAPS-NEXT: Stack Maps:   has 4 locations
; STACKMAPS-NEXT: Stack Maps: 		Loc 0: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 1: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 2: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 		Loc 3: Constant 16	[encoding: .byte 4, .byte 8, .short 0, .int 16]
; STACKMAPS-NEXT: Stack Maps: 	has 0 live-out registers
; STACKMAPS-NEXT: Stack Maps: callsite 2882400015
; STACKMAPS-NEXT: Stack Maps:   has 4 locations
; STACKMAPS-NEXT: Stack Maps: 		Loc 0: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 1: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 2: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 		Loc 3: Constant 2	[encoding: .byte 4, .byte 8, .short 0, .int 2]
; STACKMAPS-NEXT: Stack Maps: 	has 0 live-out registers
; STACKMAPS-NEXT: Stack Maps: callsite 2882400015
; STACKMAPS-NEXT: Stack Maps:   has 4 locations
; STACKMAPS-NEXT: Stack Maps: 		Loc 0: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 1: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 2: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 		Loc 3: Constant 3	[encoding: .byte 4, .byte 8, .short 0, .int 3]
; STACKMAPS-NEXT: Stack Maps: 	has 0 live-out registers
; STACKMAPS-NEXT: Stack Maps: callsite 4243
; STACKMAPS-NEXT: Stack Maps:   has 4 locations
; STACKMAPS-NEXT: Stack Maps: 		Loc 0: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 1: Constant 0	[encoding: .byte 4, .byte 8, .short 0, .int 0]
; STACKMAPS-NEXT: Stack Maps: 		Loc 2: Constant 1	[encoding: .byte 4, .byte 8, .short 0, .int 1]
; STACKMAPS-NEXT: Stack Maps: 		Loc 3: Constant 55	[encoding: .byte 4, .byte 8, .short 0, .int 55]
; STACKMAPS-NEXT: Stack Maps: 	has 0 live-out registers


declare i32 @callee_0()
declare i32 @callee_1(i32)
declare i32 @callee_vararg(...)

define i32 @caller_0() {
; CHECK-LABEL: _caller_0
entry:
  %v = call i32 @callee_0() [ "deopt"(i32 0) ]
  %v2 = add i32 %v, 1
  ret i32 %v2
; CHECK:	callq	_callee_0
; CHECK:	incl	%eax
; CHECK:	retq
}

define i32 @caller_1() {
; CHECK-LABEL: _caller_1
entry:
  %v = call i32 @callee_1(i32 42) "statepoint-id"="4242" [ "deopt"(i32 1) ]
  ret i32 %v
; CHECK:	callq	_callee_1
; CHECK:	popq	%rcx
; CHECK:	retq
}

define i32 @caller_vararg() {
; CHECK-LABEL: _caller_vararg
entry:
; CHECK: movb    $1, %al
; CHECK: callq   _callee_vararg
  %v = call i32(...) @callee_vararg(i32 42, double 500.0) "statepoint-id"="4243" [ "deopt"(i32 16) ]
  ret i32 %v
}

define i32 @invoker_0() personality i8 0 {
; CHECK-LABEL: _invoker_0
entry:
  %v = invoke i32 @callee_0() [ "deopt"(i32 2) ]
          to label %normal unwind label %uw

normal:
  ret i32 %v

uw:
  %ehvals = landingpad { i8*, i32 }
      cleanup
  ret i32 1
; CHECK:	callq	_callee_0
; CHECK:	popq	%rcx
; CHECK:	retq
; CHECK:	movl	$1, %eax
; CHECK:	popq	%rcx
; CHECK:	retq
}

define i32 @invoker_1() personality i8 0 {
; CHECK-LABEL: _invoker_1
entry:
  %v = invoke i32 @callee_1(i32 45) "statepoint-num-patch-bytes"="9" [ "deopt"(i32 3) ]
          to label %normal unwind label %uw

normal:
  ret i32 %v

uw:
  %ehvals = landingpad { i8*, i32 }
      cleanup
  ret i32 1
; CHECK:	movl	$45, %edi
; CHECK:	nopw    512(%rax,%rax)
; CHECK:	popq	%rcx
; CHECK:	retq
; CHECK:	movl	$1, %eax
; CHECK:	popq	%rcx
; CHECK:	retq
}

define i32 @invoker_2() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %val = invoke i32 @callee_1(i32 1)
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:
  %cp1 = catchpad within %cs1 [i8* null, i32 64, i8* null]
  %val2 = call i32 @callee_1(i32 100) "statepoint-id"="4243" [ "funclet"(token %cp1), "deopt"(i32 55) ]
  catchret from %cp1 to label %try.cont

try.cont:
  ret i32 0
}

declare i32 @__CxxFrameHandler3(...)

define void @f_0(i64 %n) {
  ; CHECK-LABEL: _f_0
  %s = alloca i64
  %vl = alloca i64, i64 %n
  ; Check that the stackmap does not reference %s through
  ; SP since the offset is not static because of %vl.
  ; STACKMAPS: Loc 3: Direct 6
  call void @g_0(i64* %vl) [ "deopt"(i64* %s) ]
  ret void
}

declare void @g_0(i64* %vl)
