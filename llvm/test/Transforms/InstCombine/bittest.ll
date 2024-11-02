; RUN: opt < %s -instcombine -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S |\
; RUN:    not grep "call void @abort"

@b_rec.0 = external global i32          ; <ptr> [#uses=2]

define void @_Z12h000007_testv(ptr %P) {
entry:
        %tmp.2 = load i32, ptr @b_rec.0             ; <i32> [#uses=1]
        %tmp.9 = or i32 %tmp.2, -989855744              ; <i32> [#uses=2]
        %tmp.16 = and i32 %tmp.9, -805306369            ; <i32> [#uses=2]
        %tmp.17 = and i32 %tmp.9, -973078529            ; <i32> [#uses=1]
        store i32 %tmp.17, ptr @b_rec.0
        %tmp.17.shrunk = bitcast i32 %tmp.16 to i32             ; <i32> [#uses=1]
        %tmp.22 = and i32 %tmp.17.shrunk, -1073741824           ; <i32> [#uses=1]
        %tmp.23 = icmp eq i32 %tmp.22, -1073741824              ; <i1> [#uses=1]
        br i1 %tmp.23, label %endif.0, label %then.0

then.0:         ; preds = %entry
        tail call void @abort( )
        unreachable

endif.0:                ; preds = %entry
        %tmp.17.shrunk2 = bitcast i32 %tmp.16 to i32            ; <i32> [#uses=1]
        %tmp.27.mask = and i32 %tmp.17.shrunk2, 100663295               ; <i32> [#uses=1]
        store i32 %tmp.27.mask, ptr %P
        ret void
}

declare void @abort()

