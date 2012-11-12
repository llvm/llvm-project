target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-montavista-linux-gnuspe"
; RUN: llc < %s -march=ppc32 | FileCheck %s

%struct.__va_list_tag.0.9.18.23.32.41.48.55.62.67.72.77.82.87.90.93.96.101.105 = type { i8, i8, i16, i8*, i8* }

define fastcc void @test1(%struct.__va_list_tag.0.9.18.23.32.41.48.55.62.67.72.77.82.87.90.93.96.101.105* %args) {
entry:
  br i1 undef, label %repeat, label %maxlen_reached

repeat:                                           ; preds = %entry
  switch i32 undef, label %sw.bb323 [
    i32 77, label %sw.bb72
    i32 111, label %sw.bb309
    i32 80, label %sw.bb313
    i32 117, label %sw.bb326
    i32 88, label %sw.bb321
  ]

sw.bb72:                                          ; preds = %repeat
  unreachable

sw.bb309:                                         ; preds = %repeat
  unreachable

sw.bb313:                                         ; preds = %repeat
  unreachable

sw.bb321:                                         ; preds = %repeat
  unreachable

sw.bb323:                                         ; preds = %repeat
  %0 = va_arg %struct.__va_list_tag.0.9.18.23.32.41.48.55.62.67.72.77.82.87.90.93.96.101.105* %args, i32
  unreachable

sw.bb326:                                         ; preds = %repeat
  unreachable

maxlen_reached:                                   ; preds = %entry
  ret void
}

; If the SD nodes are not cleaup up correctly, then this can fail to compile
; with an error like:  Cannot select: ch = setlt [ID=6]
; CHECK: @test1

