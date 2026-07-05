; This is the sequence of stuff that the Java front-end expands for a single 
; <= comparison.  Check to make sure we turn it into a <= (only)

; RUN: opt < %s -passes=instcombine -S | grep "icmp sle i32 %A, %B"

define i1 @le(i32 %A, i32 %B) {
        %c1 = icmp sgt i32 %A, %B               ; <i1> [#uses=1]
        %tmp = select i1 %c1, i32 1, i32 0              ; <i32> [#uses=1]
        %c2 = icmp slt i32 %A, %B               ; <i1> [#uses=1]
        %result = select i1 %c2, i32 -1, i32 %tmp               ; <i32> [#uses=1]
        %c3 = icmp sle i32 %result, 0           ; <i1> [#uses=1]
        ret i1 %c3
}

