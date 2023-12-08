; RUN: llc < %s

;;
;; Test the sequence:
;;	cast -> setle 0, %cast -> br %cond
;; This sequence should cause the cast value to be forwarded twice,
;; i.e., cast is forwarded to the setle and the setle is forwarded
;; to the branch.
;; register argument of the "branch-on-register" instruction, i.e.,
;; 
;; This produces the bogus output instruction:
;;	brlez   <NULL VALUE>, .L_SumArray_bb3.
;; This came from %bb1 of sumarrray.ll generated from sumarray.c.

define i32 @SumArray(i32 %Num) {
        %Num.upgrd.1 = alloca i32               ; <ptr> [#uses=2]
        br label %Top

Top:            ; preds = %Top, %0
        store i32 %Num, ptr %Num.upgrd.1
        %reg108 = load i32, ptr %Num.upgrd.1                ; <i32> [#uses=1]
        %cast1006 = bitcast i32 %reg108 to i32          ; <i32> [#uses=1]
        %cond1001 = icmp ule i32 %cast1006, 0           ; <i1> [#uses=1]
        br i1 %cond1001, label %bb6, label %Top

bb6:            ; preds = %Top
        ret i32 42
}

