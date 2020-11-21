; FIXME: Add tests for global-isel/fast-isel.

; RUN: llc < %s -mtriple=arm64-windows | FileCheck %s

; Returns <= 8 bytes should be in X0.
%struct.S1 = type { i32, i32 }
define dso_local i64 @"?f1"() {
entry:
; CHECK-LABEL: f1
; CHECK: str xzr, [sp, #8]
; CHECK: mov x0, xzr

  %retval = alloca %struct.S1, align 4
  %a = getelementptr inbounds %struct.S1, %struct.S1* %retval, i32 0, i32 0
  store i32 0, i32* %a, align 4
  %b = getelementptr inbounds %struct.S1, %struct.S1* %retval, i32 0, i32 1
  store i32 0, i32* %b, align 4
  %0 = bitcast %struct.S1* %retval to i64*
  %1 = load i64, i64* %0, align 4
  ret i64 %1
}

; Returns <= 16 bytes should be in X0/X1.
%struct.S2 = type { i32, i32, i32, i32 }
define dso_local [2 x i64] @"?f2"() {
entry:
; FIXME: Missed optimization, the entire SP push/pop could be removed
; CHECK-LABEL: f2
; CHECK:         sub     sp, sp, #16
; CHECK-NEXT:    .seh_stackalloc 16
; CHECK-NEXT:    .seh_endprologue
; CHECK-NEXT:    stp     xzr, xzr, [sp]
; CHECK-NEXT:    mov     x0, xzr
; CHECK-NEXT:    mov     x1, xzr
; CHECK-NEXT:    .seh_startepilogue
; CHECK-NEXT:    add     sp, sp, #16

  %retval = alloca %struct.S2, align 4
  %a = getelementptr inbounds %struct.S2, %struct.S2* %retval, i32 0, i32 0
  store i32 0, i32* %a, align 4
  %b = getelementptr inbounds %struct.S2, %struct.S2* %retval, i32 0, i32 1
  store i32 0, i32* %b, align 4
  %c = getelementptr inbounds %struct.S2, %struct.S2* %retval, i32 0, i32 2
  store i32 0, i32* %c, align 4
  %d = getelementptr inbounds %struct.S2, %struct.S2* %retval, i32 0, i32 3
  store i32 0, i32* %d, align 4
  %0 = bitcast %struct.S2* %retval to [2 x i64]*
  %1 = load [2 x i64], [2 x i64]* %0, align 4
  ret [2 x i64] %1
}

; Arguments > 16 bytes should be passed in X8.
%struct.S3 = type { i32, i32, i32, i32, i32 }
define dso_local void @"?f3"(%struct.S3* noalias sret(%struct.S3) %agg.result) {
entry:
; CHECK-LABEL: f3
; CHECK: stp xzr, xzr, [x8]
; CHECK: str wzr, [x8, #16]

  %a = getelementptr inbounds %struct.S3, %struct.S3* %agg.result, i32 0, i32 0
  store i32 0, i32* %a, align 4
  %b = getelementptr inbounds %struct.S3, %struct.S3* %agg.result, i32 0, i32 1
  store i32 0, i32* %b, align 4
  %c = getelementptr inbounds %struct.S3, %struct.S3* %agg.result, i32 0, i32 2
  store i32 0, i32* %c, align 4
  %d = getelementptr inbounds %struct.S3, %struct.S3* %agg.result, i32 0, i32 3
  store i32 0, i32* %d, align 4
  %e = getelementptr inbounds %struct.S3, %struct.S3* %agg.result, i32 0, i32 4
  store i32 0, i32* %e, align 4
  ret void
}

; InReg arguments to non-instance methods must be passed in X0 and returns in
; X0.
%class.B = type { i32 }
define dso_local void @"?f4"(%class.B* inreg noalias nocapture sret(%class.B) %agg.result) {
entry:
; CHECK-LABEL: f4
; CHECK: mov w8, #1
; CHECK: str w8, [x0]
  %X.i = getelementptr inbounds %class.B, %class.B* %agg.result, i64 0, i32 0
  store i32 1, i32* %X.i, align 4
  ret void
}

; InReg arguments to instance methods must be passed in X1 and returns in X0.
%class.C = type { i8 }
%class.A = type { i8 }

define dso_local void @"?inst@C"(%class.C* %this, %class.A* inreg noalias sret(%class.A) %agg.result) {
entry:
; CHECK-LABEL: inst@C
; CHECK: str x0, [sp, #8]
; CHECK: mov x0, x1

  %this.addr = alloca %class.C*, align 8
  store %class.C* %this, %class.C** %this.addr, align 8
  %this1 = load %class.C*, %class.C** %this.addr, align 8
  ret void
}
