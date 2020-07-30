; RUN: opt < %s -loop-reduce -S | FileCheck %s
; RUN: opt -passes='require<scalar-evolution>,require<targetir>,loop(loop-reduce)' < %s -S | FileCheck %s
;
; PR11782: bad cast to AddRecExpr.
; A sign extend feeds an IVUser and cannot be hoisted into the AddRec.
; CollectIVChains should bailout on this case.


; Provide legal integer types.
target datalayout = "n8:16:32:64"

%struct = type { i8*, i8*, i16, i64, i16, i16, i16, i64, i64, i16, i8*, i64, i64, i64 }

; CHECK-LABEL: @test(
; CHECK: for.body:
; CHECK: lsr.iv = phi %struct
; CHECK: br
define i32 @test(i8* %h, i32 %more) nounwind uwtable {
entry:
  br i1 undef, label %land.end238, label %return

land.end238:                                      ; preds = %if.end229
  br label %for.body

for.body:                                         ; preds = %sw.epilog, %land.end238
  %fbh.0 = phi %struct* [ undef, %land.end238 ], [ %incdec.ptr, %sw.epilog ]
  %column_n.0 = phi i16 [ 0, %land.end238 ], [ %inc601, %sw.epilog ]
  %conv250 = sext i16 %column_n.0 to i32
  %add257 = add nsw i32 %conv250, 1
  %conv258 = trunc i32 %add257 to i16
  %cmp263 = icmp ult i16 undef, 2
  br label %if.end388

if.end388:                                        ; preds = %if.then380, %if.else356
  %ColLength = getelementptr inbounds %struct, %struct* %fbh.0, i64 0, i32 7
  %call405 = call signext i16 @SQLColAttribute(i8* undef, i16 zeroext %conv258, i16 zeroext 1003, i8* null, i16 signext 0, i16* null, i64* %ColLength) nounwind
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb542, %sw.bb523, %if.end475
  %inc601 = add i16 %column_n.0, 1
  %incdec.ptr = getelementptr inbounds %struct, %struct* %fbh.0, i64 1
  br label %for.body

return:                                           ; preds = %entry
  ret i32 1
}

declare signext i16 @SQLColAttribute(i8*, i16 zeroext, i16 zeroext, i8*, i16 signext, i16*, i64*)
