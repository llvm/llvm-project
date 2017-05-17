; RUN: llc < %s -mtriple=thumbv7-apple-darwin -arm-atomic-cfg-tidy=0 -mcpu=cortex-a8 -relocation-model=pic -disable-fp-elim | FileCheck %s
; rdar://7352504
; Make sure we use "str r9, [sp, #+28]" instead of "sub.w r4, r7, #256" followed by "str r9, [r4, #-32]".

%0 = type { i16, i8, i8 }
%1 = type { [2 x i32], [2 x i32] }
%2 = type { %union.rec* }
%struct.FILE_POS = type { i8, i8, i16, i32 }
%struct.GAP = type { i8, i8, i16 }
%struct.LIST = type { %union.rec*, %union.rec* }
%struct.STYLE = type { %union.anon, %union.anon, i16, i16, i32 }
%struct.head_type = type { [2 x %struct.LIST], %union.FIRST_UNION, %union.SECOND_UNION, %union.THIRD_UNION, %union.FOURTH_UNION, %union.rec*, %2, %union.rec*, %union.rec*, %union.rec*, %union.rec*, %union.rec*, %union.rec*, %union.rec*, %union.rec*, i32 }
%union.FIRST_UNION = type { %struct.FILE_POS }
%union.FOURTH_UNION = type { %struct.STYLE }
%union.SECOND_UNION = type { %0 }
%union.THIRD_UNION = type { %1 }
%union.anon = type { %struct.GAP }
%union.rec = type { %struct.head_type }

@zz_hold = external global %union.rec*            ; <%union.rec**> [#uses=2]
@zz_res = external global %union.rec*             ; <%union.rec**> [#uses=1]

define %union.rec* @Manifest(%union.rec* %x, %union.rec* %env, %struct.STYLE* %style, %union.rec** %bthr, %union.rec** %fthr, %union.rec** %target, %union.rec** %crs, i32 %ok, i32 %need_expand, %union.rec** %enclose, i32 %fcr) nounwind {
entry:
; CHECK:       ldr{{(.w)?}}	{{(r[0-9]+)|(lr)}}, [r7, #28]
  %xgaps.i = alloca [32 x %union.rec*], align 4   ; <[32 x %union.rec*]*> [#uses=0]
  %ycomp.i = alloca [32 x %union.rec*], align 4   ; <[32 x %union.rec*]*> [#uses=0]
  br label %bb20

bb20:                                             ; preds = %entry
  switch i32 undef, label %bb1287 [
    i32 110, label %bb119
    i32 120, label %bb119
    i32 210, label %bb420
    i32 230, label %bb420
    i32 450, label %bb438
    i32 460, label %bb438
    i32 550, label %bb533
    i32 560, label %bb569
    i32 640, label %bb745
    i32 780, label %bb1098
  ]

bb119:                                            ; preds = %bb20, %bb20
  unreachable

bb420:                                            ; preds = %bb20, %bb20
; CHECK: bb420
; CHECK: str{{(.w)?}} r{{[0-9]+}}, [sp
; CHECK: str{{(.w)?}} r{{[0-9]+}}, [sp
; CHECK: str{{(.w)?}} r{{[0-9]+}}, [sp
; CHECK: str{{(.w)?}} r{{[0-9]+}}, [sp
  store volatile %union.rec* null, %union.rec** @zz_hold, align 4
  store %union.rec* null, %union.rec** @zz_res, align 4
  store volatile %union.rec* %x, %union.rec** @zz_hold, align 4
  %0 = call  %union.rec* @Manifest(%union.rec* undef, %union.rec* %env, %struct.STYLE* %style, %union.rec** %bthr, %union.rec** %fthr, %union.rec** %target, %union.rec** %crs, i32 %ok, i32 %need_expand, %union.rec** %enclose, i32 %fcr) nounwind ; <%union.rec*> [#uses=0]
  unreachable

bb438:                                            ; preds = %bb20, %bb20
  unreachable

bb533:                                            ; preds = %bb20
  ret %union.rec* %x

bb569:                                            ; preds = %bb20
  unreachable

bb745:                                            ; preds = %bb20
  unreachable

bb1098:                                           ; preds = %bb20
  unreachable

bb1287:                                           ; preds = %bb20
  unreachable
}
