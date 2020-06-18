; RUN: llc -verify-machineinstrs -code-model=small < %s | FileCheck %s
; Test with null streamer.
; RUN: llc -O0 -filetype=null < %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@double_array = global [32 x double] zeroinitializer, align 8
@number64 = global i64 10, align 8
@internal_static_var.x = internal unnamed_addr global i64 0, align 8

define i64 @access_int64(i64 %a) nounwind readonly {
entry:
; CHECK-LABEL: access_int64:
; CHECK-NEXT: .p2align  3
; CHECK-NEXT: .quad   .L[[BEGIN:.*]]
; CHECK-NEXT: .quad   .TOC.@tocbase
; CHECK-NEXT: .quad   0
; CHECK-NEXT: .text
; CHECK-NEXT: .L[[BEGIN]]:
  %0 = load i64, i64* @number64, align 8
; CHECK: ld {{[0-9]+}}, .LC{{[0-9]+}}@toc(2)
  %cmp = icmp eq i64 %0, %a
  %conv1 = zext i1 %cmp to i64 
  ret i64 %conv1
}

define i64 @internal_static_var(i64 %a) nounwind {
entry:
; CHECK-LABEL: internal_static_var:
; CHECK: ld {{[0-9]+}}, .LC{{[0-9]+}}@toc(2)
  %0 = load i64, i64* @internal_static_var.x, align 8
  %cmp = icmp eq i64 %0, %a
  %conv1 = zext i1 %cmp to i64 
  ret i64 %conv1 
}

define i32 @access_double(double %a) nounwind readnone {
entry:
; CHECK-LABEL: access_double:
; CHECK: ld {{[0-9]+}}, .LC{{[0-9]+}}@toc(2)
  %cmp = fcmp oeq double %a, 2.000000e+00
  %conv = zext i1 %cmp to i32 
  ret i32 %conv
}


define i32 @access_double_array(double %a, i32 %i) nounwind readonly {
entry:
; CHECK-LABEL: access_double_array:
  %idxprom = sext i32 %i to i64
  %arrayidx = getelementptr inbounds [32 x double], [32 x double]* @double_array, i64 0, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
; CHECK: ld {{[0-9]+}}, .LC{{[0-9]+}}@toc(2)
  %cmp = fcmp oeq double %0, %a
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; Check the creation of 4 .tc entries:
; * int64_t global 'number64'
; * double constant 2.0
; * double array 'double_array'
; * static int64_t 'x' accessed within '@internal_static_var'
; CHECK: .LC{{[0-9]+}}:
; CHECK-NEXT: .tc {{[\._a-zA-Z0-9]+}}[TC],{{[\._a-zA-Z0-9]+}}
; CHECK-NEXT: .LC{{[0-9]+}}:
; CHECK-NEXT: .tc {{[\._a-zA-Z0-9]+}}[TC],{{[\._a-zA-Z0-9]+}}
; CHECK-NEXT: .LC{{[0-9]+}}:
; CHECK-NEXT: .tc {{[\._a-zA-Z0-9]+}}[TC],{{[\._a-zA-Z0-9]+}}
; CHECK-NEXT: .LC{{[0-9]+}}:
; CHECK-NEXT: .tc {{[\._a-zA-Z0-9]+}}[TC],{{[\._a-zA-Z0-9]+}}
