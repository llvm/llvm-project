; RUN: llc < %s -o /dev/null
; rdar://10795250
; DAGCombiner should converge.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64"
target triple = "arm64-apple-macosx10.8.0"

define i64 @foo(i128 %Params.coerce, i128 %SelLocs.coerce) {
entry:
  %tmp = lshr i128 %Params.coerce, 61
  %.tr38.i = trunc i128 %tmp to i64
  %mul.i = and i64 %.tr38.i, 4294967288
  %tmp1 = lshr i128 %SelLocs.coerce, 62
  %.tr.i = trunc i128 %tmp1 to i64
  %mul7.i = and i64 %.tr.i, 4294967292
  %add.i = add i64 %mul7.i, %mul.i
  %conv.i.i = and i64 %add.i, 4294967292
  ret i64 %conv.i.i
}
