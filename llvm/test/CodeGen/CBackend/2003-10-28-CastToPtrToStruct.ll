; RUN: llvm-as < %s | llc -march=c

; reduced from DOOM.
        %union._XEvent = type { i32 }
@.X_event_9 = global %union._XEvent zeroinitializer             ; <%union._XEvent*> [#uses=1]

define void @I_InitGraphics() {
shortcirc_next.3:
        %tmp.319 = load i32* getelementptr ({ i32, i32 }* bitcast (%union._XEvent* @.X_event_9 to { i32, i32 }*), i64 0, i32 1)               ; <i32> [#uses=0]
        ret void
}

