; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s

; CHECK: pr13209:
; CHECK-NOT: mov
; CHECK: .size pr13209

define zeroext i1 @pr13209(i8** %x, i8*** %jumpTable) nounwind {
if.end51:
  br label %indirectgoto.preheader
indirectgoto.preheader:
  %frombool.i5915.ph = phi i8 [ undef, %if.end51 ], [ %frombool.i5917, %jit_return ]
  br label %indirectgoto
do.end165:
  %tmp92 = load i8** %x, align 8
  br label %indirectgoto
do.end209:
  %tmp104 = load i8** %x, align 8
  br label %indirectgoto
do.end220:
  %tmp107 = load i8** %x, align 8
  br label %indirectgoto
do.end231:
  %tmp110 = load i8** %x, align 8
  br label %indirectgoto
do.end242:
  %tmp113 = load i8** %x, align 8
  br label %indirectgoto
do.end253:
  %tmp116 = load i8** %x, align 8
  br label %indirectgoto
do.end286:
  %tmp125 = load i8** %x, align 8
  br label %indirectgoto
do.end297:
  %tmp128 = load i8** %x, align 8
  br label %indirectgoto
do.end308:
  %tmp131 = load i8** %x, align 8
  br label %indirectgoto
do.end429:
  %tmp164 = load i8** %x, align 8
  br label %indirectgoto
do.end440:
  %tmp167 = load i8** %x, align 8
  br label %indirectgoto
do.body482:
  br i1 false, label %indirectgoto, label %do.body495
do.body495:
  br label %indirectgoto
do.end723:
  br label %inline_return
inline_return:
  %frombool.i5917 = phi i8 [ 0, %if.end5571 ], [ %frombool.i5915, %do.end723 ]
  br label %jit_return
jit_return:
  br label %indirectgoto.preheader
L_JSOP_UINT24:
  %tmp864 = load i8** %x, align 8
  br label %indirectgoto
L_JSOP_THROWING:
  %tmp1201 = load i8** %x, align 8
  br label %indirectgoto
do.body4936:
  %tmp1240 = load i8** %x, align 8
  br label %indirectgoto
do.body5184:
  %tmp1340 = load i8** %x, align 8
  br label %indirectgoto
if.end5571:
  br  label %inline_return
indirectgoto:
  %frombool.i5915 = phi i8  [ 0, %do.body495 ],[ 0, %do.body482 ] , [ %frombool.i5915, %do.body4936 ],[ %frombool.i5915, %do.body5184 ], [ %frombool.i5915, %L_JSOP_UINT24 ], [ %frombool.i5915, %do.end286 ], [ %frombool.i5915, %do.end297 ], [ %frombool.i5915, %do.end308 ], [ %frombool.i5915, %do.end429 ], [ %frombool.i5915, %do.end440 ], [ %frombool.i5915, %L_JSOP_THROWING ], [ %frombool.i5915, %do.end253 ], [ %frombool.i5915, %do.end242 ], [ %frombool.i5915, %do.end231 ], [ %frombool.i5915, %do.end220 ], [ %frombool.i5915, %do.end209 ],[ %frombool.i5915, %do.end165 ], [ %frombool.i5915.ph, %indirectgoto.preheader ]
  indirectbr i8* null, [ label %if.end5571, label %do.end165, label %do.end209, label %do.end220, label %do.end231, label %do.end242, label %do.end253, label %do.end723, label %L_JSOP_THROWING, label %do.end440, label %do.end429, label %do.end308, label %do.end297, label %do.end286, label %L_JSOP_UINT24, label %do.body5184, label %do.body4936, label %do.body482]
}
