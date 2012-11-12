; RUN: opt < %s -globaldce
;
define internal void @func() {
        ret void
}

define void @main() {
        %X = bitcast void ()* @func to i32*             ; <i32*> [#uses=0]
        ret void
}

