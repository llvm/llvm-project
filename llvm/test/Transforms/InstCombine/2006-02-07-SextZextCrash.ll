; RUN: llvm-as < %s | opt -instcombine -disable-output	

        %struct.rtx_const = type { i32, { %union.real_extract } }
        %struct.rtx_def = type { i32, [1 x %union.rtunion_def] }
        %union.real_extract = type { double }
        %union.rtunion_def = type { i32 }

define fastcc void @decode_rtx_const(%struct.rtx_def* %x, %struct.rtx_const* %value) {
        %tmp.54 = getelementptr %struct.rtx_const* %value, i32 0, i32 0         ; <i32*> [#uses=1]
        %tmp.56 = getelementptr %struct.rtx_def* %x, i32 0, i32 0               ; <i32*> [#uses=1]
        %tmp.57 = load i32* %tmp.56             ; <i32> [#uses=1]
        %tmp.58 = shl i32 %tmp.57, 8            ; <i32> [#uses=1]
        %tmp.59 = ashr i32 %tmp.58, 24          ; <i32> [#uses=1]
        %tmp.60 = trunc i32 %tmp.59 to i16              ; <i16> [#uses=1]
        %tmp.61 = zext i16 %tmp.60 to i32               ; <i32> [#uses=1]
        %tmp.62 = shl i32 %tmp.61, 16           ; <i32> [#uses=1]
        %tmp.65 = or i32 0, %tmp.62             ; <i32> [#uses=1]
        store i32 %tmp.65, i32* %tmp.54
        ret void
}

