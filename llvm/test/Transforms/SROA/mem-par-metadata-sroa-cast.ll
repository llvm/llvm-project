; RUN: opt < %s -passes=sroa -S | FileCheck %s
;
; Make sure the llvm.access.group meta-data is preserved
; when a load/store is replaced with another load/store by sroa
; Ensure this is done for casting too.
;
; CHECK: entry:
; CHECK: load i32, i32* {{.*}}, !llvm.access.group [[DISTINCT:![0-9]*]]
; CHECK: load i32, i32* {{.*}}, !llvm.access.group [[DISTINCT]]
; CHECK: ret void
; CHECK: [[DISTINCT]] = distinct !{}

%CMPLX = type { float, float }

define dso_local void @test() {
entry:
  %PART = alloca %CMPLX, align 8
  %PREV = alloca %CMPLX, align 8
  %r2 = getelementptr %CMPLX, %CMPLX* %PREV, i32 0, i32 0
  store float 0.000000e+00, float* %r2, align 4
  %i2 = getelementptr %CMPLX, %CMPLX* %PREV, i32 0, i32 1
  store float 0.000000e+00, float* %i2, align 4
  %dummy = sext i16 0 to i64
  %T = getelementptr %CMPLX, %CMPLX* %PART, i64 %dummy
  %X35 = bitcast %CMPLX* %T to i64*
  %X36 = bitcast %CMPLX* %PREV to i64*
  %X37 = load i64, i64* %X35, align 8, !llvm.access.group !0
  store i64 %X37, i64* %X36, align 8
  ret void
}

!0 = distinct !{}
