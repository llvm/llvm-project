; RUN: llc -mcpu=pwr7 -O0 < %s | FileCheck %s

; This tests receiving and re-passing parameters consisting of structures
; of size 3, 5, 6, and 7.  They are to be found/placed right-adjusted in
; the parameter registers.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.S3 = type { [3 x i8] }
%struct.S5 = type { [5 x i8] }
%struct.S6 = type { [6 x i8] }
%struct.S7 = type { [7 x i8] }

define void @test(%struct.S3* byval %s3, %struct.S5* byval %s5, %struct.S6* byval %s6, %struct.S7* byval %s7) nounwind {
entry:
  call void @check(%struct.S3* byval %s3, %struct.S5* byval %s5, %struct.S6* byval %s6, %struct.S7* byval %s7)
  ret void
}

; CHECK: std 6, 216(1)
; CHECK: std 5, 208(1)
; CHECK: std 4, 200(1)
; CHECK: std 3, 192(1)
; CHECK: lbz {{[0-9]+}}, 199(1)
; CHECK: stb {{[0-9]+}}, 55(1)
; CHECK: lhz {{[0-9]+}}, 197(1)
; CHECK: sth {{[0-9]+}}, 53(1)
; CHECK: lbz {{[0-9]+}}, 207(1)
; CHECK: stb {{[0-9]+}}, 63(1)
; CHECK: lwz {{[0-9]+}}, 203(1)
; CHECK: stw {{[0-9]+}}, 59(1)
; CHECK: lhz {{[0-9]+}}, 214(1)
; CHECK: sth {{[0-9]+}}, 70(1)
; CHECK: lwz {{[0-9]+}}, 210(1)
; CHECK: stw {{[0-9]+}}, 66(1)
; CHECK: lbz {{[0-9]+}}, 223(1)
; CHECK: stb {{[0-9]+}}, 79(1)
; CHECK: lhz {{[0-9]+}}, 221(1)
; CHECK: sth {{[0-9]+}}, 77(1)
; CHECK: lwz {{[0-9]+}}, 217(1)
; CHECK: stw {{[0-9]+}}, 73(1)
; CHECK: ld 6, 72(1)
; CHECK: ld 5, 64(1)
; CHECK: ld 4, 56(1)
; CHECK: ld 3, 48(1)

declare void @check(%struct.S3* byval, %struct.S5* byval, %struct.S6* byval, %struct.S7* byval)
