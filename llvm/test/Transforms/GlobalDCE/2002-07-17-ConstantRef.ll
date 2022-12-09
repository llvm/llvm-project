; RUN: opt < %s -passes=globaldce
;

@X = global ptr @func              ; <ptr> [#uses=0]

; Not dead, can be reachable via X
define internal void @func() {
        ret void
}

define void @main() {
        ret void
}
