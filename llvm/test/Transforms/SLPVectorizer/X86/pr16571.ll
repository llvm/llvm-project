; RUN: opt < %s -passes=slp-vectorizer -S -mtriple=i686-pc-win32 -mcpu=corei7-avx

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S32"
target triple = "i686-pc-win32"

define hidden fastcc void @"System.PrimitiveTypesParser.TryParseIEEE754<char>(char*,uint,double&)"() unnamed_addr {
"@0":
  br i1 undef, label %"@38.lr.ph", label %"@37"

"@37":                                            ; preds = %"@38.lr.ph", %"@44", %"@0"
  ret void

"@44":                                            ; preds = %"@38.lr.ph"
  %0 = add i64 undef, undef
  %1 = add i32 %mainPartDigits.loc.0.ph45, 1
  br i1 undef, label %"@38.lr.ph", label %"@37"

"@38.lr.ph":                                      ; preds = %"@44", %"@0"
  %mainDoublePart.loc.0.ph46 = phi i64 [ %0, %"@44" ], [ 0, %"@0" ]
  %mainPartDigits.loc.0.ph45 = phi i32 [ %1, %"@44" ], [ 0, %"@0" ]
  br i1 undef, label %"@44", label %"@37"
}
