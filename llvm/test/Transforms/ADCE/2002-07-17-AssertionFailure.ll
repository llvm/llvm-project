; This testcase fails because ADCE does not correctly delete the chain of 
; three instructions that are dead here.  Ironically there were a dead basic
; block in this function, it would work fine, but that would be the part we 
; have to fix now, wouldn't it....
;
; RUN: opt < %s -adce

define void @foo(i8* %reg5481) {
        %cast611 = bitcast i8* %reg5481 to i8**         ; <i8**> [#uses=1]
        %reg162 = load i8** %cast611            ; <i8*> [#uses=1]
        ptrtoint i8* %reg162 to i32             ; <i32>:1 [#uses=0]
        ret void
}
