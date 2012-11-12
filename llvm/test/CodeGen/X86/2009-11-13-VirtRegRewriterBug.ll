; RUN: llc < %s -mtriple=i386-apple-darwin -relocation-model=pic -disable-fp-elim
; rdar://7394770

%struct.JVTLib_100487 = type <{ i8 }>

define i32 @_Z13JVTLib_10335613JVTLib_10266513JVTLib_100579S_S_S_jPhj(i16* nocapture %ResidualX_Array.0, %struct.JVTLib_100487* nocapture byval align 4 %xqp, i16* nocapture %ResidualL_Array.0, i16* %ResidualDCZ_Array.0, i16* nocapture %ResidualACZ_FOArray.0, i32 %useFRextDequant, i8* nocapture %JVTLib_103357, i32 %use_field_scan) ssp {
bb.nph:
  %0 = shl i32 undef, 1                           ; <i32> [#uses=2]
  %mask133.masked.masked.masked.masked.masked.masked = or i640 undef, undef ; <i640> [#uses=1]
  br label %bb

bb:                                               ; preds = %_ZL13JVTLib_105204PKsPK13JVTLib_105184PsPhjS5_j.exit, %bb.nph
  br i1 undef, label %bb2, label %bb1

bb1:                                              ; preds = %bb
  br i1 undef, label %bb.i, label %bb1.i

bb2:                                              ; preds = %bb
  unreachable

bb.i:                                             ; preds = %bb1
  br label %_ZL13JVTLib_105204PKsPK13JVTLib_105184PsPhjS5_j.exit

bb1.i:                                            ; preds = %bb1
  br label %_ZL13JVTLib_105204PKsPK13JVTLib_105184PsPhjS5_j.exit

_ZL13JVTLib_105204PKsPK13JVTLib_105184PsPhjS5_j.exit: ; preds = %bb1.i, %bb.i
  br i1 undef, label %bb5, label %bb

bb5:                                              ; preds = %_ZL13JVTLib_105204PKsPK13JVTLib_105184PsPhjS5_j.exit
  %mask271.masked.masked.masked.masked.masked.masked.masked = or i256 0, undef ; <i256> [#uses=2]
  %mask266.masked.masked.masked.masked.masked.masked = or i256 %mask271.masked.masked.masked.masked.masked.masked.masked, undef ; <i256> [#uses=1]
  %mask241.masked = or i256 undef, undef          ; <i256> [#uses=1]
  %ins237 = or i256 undef, 0                      ; <i256> [#uses=1]
  br i1 undef, label %bb9, label %bb10

bb9:                                              ; preds = %bb5
  br i1 undef, label %bb12.i, label %_ZL13JVTLib_105255PKsPK13JVTLib_105184Psj.exit

bb12.i:                                           ; preds = %bb9
  br label %_ZL13JVTLib_105255PKsPK13JVTLib_105184Psj.exit

_ZL13JVTLib_105255PKsPK13JVTLib_105184Psj.exit:   ; preds = %bb12.i, %bb9
  ret i32 undef

bb10:                                             ; preds = %bb5
  %1 = sext i16 undef to i32                      ; <i32> [#uses=1]
  %2 = sext i16 undef to i32                      ; <i32> [#uses=1]
  %3 = sext i16 undef to i32                      ; <i32> [#uses=1]
  %4 = sext i16 undef to i32                      ; <i32> [#uses=1]
  %5 = sext i16 undef to i32                      ; <i32> [#uses=1]
  %6 = sext i16 undef to i32                      ; <i32> [#uses=1]
  %tmp211 = lshr i256 %mask271.masked.masked.masked.masked.masked.masked.masked, 112 ; <i256> [#uses=0]
  %7 = sext i16 undef to i32                      ; <i32> [#uses=1]
  %tmp208 = lshr i256 %mask266.masked.masked.masked.masked.masked.masked, 128 ; <i256> [#uses=1]
  %tmp209 = trunc i256 %tmp208 to i16             ; <i16> [#uses=1]
  %8 = sext i16 %tmp209 to i32                    ; <i32> [#uses=1]
  %9 = sext i16 undef to i32                      ; <i32> [#uses=1]
  %10 = sext i16 undef to i32                     ; <i32> [#uses=1]
  %tmp193 = lshr i256 %mask241.masked, 208        ; <i256> [#uses=1]
  %tmp194 = trunc i256 %tmp193 to i16             ; <i16> [#uses=1]
  %11 = sext i16 %tmp194 to i32                   ; <i32> [#uses=1]
  %tmp187 = lshr i256 %ins237, 240                ; <i256> [#uses=1]
  %tmp188 = trunc i256 %tmp187 to i16             ; <i16> [#uses=1]
  %12 = sext i16 %tmp188 to i32                   ; <i32> [#uses=1]
  %13 = add nsw i32 %4, %1                        ; <i32> [#uses=1]
  %14 = add nsw i32 %5, 0                         ; <i32> [#uses=1]
  %15 = add nsw i32 %6, %2                        ; <i32> [#uses=1]
  %16 = add nsw i32 %7, %3                        ; <i32> [#uses=1]
  %17 = add nsw i32 0, %8                         ; <i32> [#uses=1]
  %18 = add nsw i32 %11, %9                       ; <i32> [#uses=1]
  %19 = add nsw i32 0, %10                        ; <i32> [#uses=1]
  %20 = add nsw i32 %12, 0                        ; <i32> [#uses=1]
  %21 = add nsw i32 %17, %13                      ; <i32> [#uses=2]
  %22 = add nsw i32 %18, %14                      ; <i32> [#uses=2]
  %23 = add nsw i32 %19, %15                      ; <i32> [#uses=2]
  %24 = add nsw i32 %20, %16                      ; <i32> [#uses=2]
  %25 = add nsw i32 %22, %21                      ; <i32> [#uses=2]
  %26 = add nsw i32 %24, %23                      ; <i32> [#uses=2]
  %27 = sub i32 %21, %22                          ; <i32> [#uses=1]
  %28 = sub i32 %23, %24                          ; <i32> [#uses=1]
  %29 = add nsw i32 %26, %25                      ; <i32> [#uses=1]
  %30 = sub i32 %25, %26                          ; <i32> [#uses=1]
  %31 = sub i32 %27, %28                          ; <i32> [#uses=1]
  %32 = ashr i32 %29, 1                           ; <i32> [#uses=2]
  %33 = ashr i32 %30, 1                           ; <i32> [#uses=2]
  %34 = ashr i32 %31, 1                           ; <i32> [#uses=2]
  %35 = icmp sgt i32 %32, 32767                   ; <i1> [#uses=1]
  %o0_0.0.i = select i1 %35, i32 32767, i32 %32   ; <i32> [#uses=2]
  %36 = icmp slt i32 %o0_0.0.i, -32768            ; <i1> [#uses=1]
  %37 = icmp sgt i32 %33, 32767                   ; <i1> [#uses=1]
  %o1_0.0.i = select i1 %37, i32 32767, i32 %33   ; <i32> [#uses=2]
  %38 = icmp slt i32 %o1_0.0.i, -32768            ; <i1> [#uses=1]
  %39 = icmp sgt i32 %34, 32767                   ; <i1> [#uses=1]
  %o2_0.0.i = select i1 %39, i32 32767, i32 %34   ; <i32> [#uses=2]
  %40 = icmp slt i32 %o2_0.0.i, -32768            ; <i1> [#uses=1]
  %tmp101 = lshr i640 %mask133.masked.masked.masked.masked.masked.masked, 256 ; <i640> [#uses=1]
  %41 = trunc i32 %o0_0.0.i to i16                ; <i16> [#uses=1]
  %tmp358 = select i1 %36, i16 -32768, i16 %41    ; <i16> [#uses=2]
  %42 = trunc i32 %o1_0.0.i to i16                ; <i16> [#uses=1]
  %tmp347 = select i1 %38, i16 -32768, i16 %42    ; <i16> [#uses=1]
  %43 = trunc i32 %o2_0.0.i to i16                ; <i16> [#uses=1]
  %tmp335 = select i1 %40, i16 -32768, i16 %43    ; <i16> [#uses=1]
  %44 = icmp sgt i16 %tmp358, -1                  ; <i1> [#uses=2]
  %..i24 = select i1 %44, i16 %tmp358, i16 undef  ; <i16> [#uses=1]
  %45 = icmp sgt i16 %tmp347, -1                  ; <i1> [#uses=1]
  %46 = icmp sgt i16 %tmp335, -1                  ; <i1> [#uses=1]
  %47 = zext i16 %..i24 to i32                    ; <i32> [#uses=1]
  %tmp = trunc i640 %tmp101 to i32                ; <i32> [#uses=1]
  %48 = and i32 %tmp, 65535                       ; <i32> [#uses=2]
  %49 = mul i32 %47, %48                          ; <i32> [#uses=1]
  %50 = zext i16 undef to i32                     ; <i32> [#uses=1]
  %51 = mul i32 %50, %48                          ; <i32> [#uses=1]
  %52 = add i32 %49, %0                           ; <i32> [#uses=1]
  %53 = add i32 %51, %0                           ; <i32> [#uses=1]
  %54 = lshr i32 %52, undef                       ; <i32> [#uses=1]
  %55 = lshr i32 %53, undef                       ; <i32> [#uses=1]
  %56 = trunc i32 %54 to i16                      ; <i16> [#uses=1]
  %57 = trunc i32 %55 to i16                      ; <i16> [#uses=1]
  %vs16Out0_0.0.i = select i1 %44, i16 %56, i16 undef ; <i16> [#uses=1]
  %vs16Out0_4.0.i = select i1 %45, i16 0, i16 undef ; <i16> [#uses=1]
  %vs16Out1_0.0.i = select i1 %46, i16 %57, i16 undef ; <i16> [#uses=1]
  br i1 undef, label %bb129.i, label %_ZL13JVTLib_105207PKsPK13JVTLib_105184Psj.exit

bb129.i:                                          ; preds = %bb10
  br label %_ZL13JVTLib_105207PKsPK13JVTLib_105184Psj.exit

_ZL13JVTLib_105207PKsPK13JVTLib_105184Psj.exit:   ; preds = %bb129.i, %bb10
  %58 = phi i16 [ %vs16Out0_4.0.i, %bb129.i ], [ undef, %bb10 ] ; <i16> [#uses=0]
  %59 = phi i16 [ undef, %bb129.i ], [ %vs16Out1_0.0.i, %bb10 ] ; <i16> [#uses=0]
  store i16 %vs16Out0_0.0.i, i16* %ResidualDCZ_Array.0, align 2
  unreachable
}
