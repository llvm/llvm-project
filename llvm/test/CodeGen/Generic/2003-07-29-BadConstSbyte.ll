; RUN: llc < %s

; Bug: PR31341
; XFAIL: avr

;; Date:     Jul 29, 2003.
;; From:     test/Programs/MultiSource/Ptrdist-bc
;; Function: ---
;; Global:   %yy_ec = internal constant [256 x sbyte] ...
;;           A subset of this array is used in the test below.
;;
;; Error:    Character '\07' was being emitted as '\a', at yy_ec[38].
;;	     When loaded, this returned the value 97 ('a'), instead of 7.
;; 
;; Incorrect LLC Output for the array yy_ec was:
;; yy_ec_1094:
;; 	.ascii	"\000\001\001\001\001\001\001\001\001\002\003\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\002\004\005\001\001\006\a\001\b\t\n\v\f\r\016\017\020\020\020\020\020\020\020\020\020\020\001\021\022\023\024\001\001\025\025\025\025\025\025\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\026\027\030\031\032\001\033\034\035\036\037 !\"#$%&'()*+,-./$0$1$234\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001\001"
;;

@yy_ec = internal constant [6 x i8] c"\06\07\01\08\01\09"               ; <[6 x i8]*> [#uses=1]
@.str_3 = internal constant [8 x i8] c"[%d] = \00"              ; <[8 x i8]*> [#uses=1]
@.str_4 = internal constant [4 x i8] c"%d\0A\00"                ; <[4 x i8]*> [#uses=1]

declare i32 @printf(i8*, ...)

define i32 @main() {
entry:
        br label %loopentry

loopentry:              ; preds = %loopentry, %entry
        %i = phi i64 [ 0, %entry ], [ %inc.i, %loopentry ]              ; <i64> [#uses=3]
        %cptr = getelementptr [6 x i8], [6 x i8]* @yy_ec, i64 0, i64 %i           ; <i8*> [#uses=1]
        %c = load i8, i8* %cptr             ; <i8> [#uses=1]
        %ignore = call i32 (i8*, ...) @printf( i8* getelementptr ([8 x i8], [8 x i8]* @.str_3, i64 0, i64 0), i64 %i )        ; <i32> [#uses=0]
        %ignore2 = call i32 (i8*, ...) @printf( i8* getelementptr ([4 x i8], [4 x i8]* @.str_4, i64 0, i64 0), i8 %c )        ; <i32> [#uses=0]
        %inc.i = add i64 %i, 1          ; <i64> [#uses=2]
        %done = icmp sle i64 %inc.i, 5          ; <i1> [#uses=1]
        br i1 %done, label %loopentry, label %exit.1

exit.1:         ; preds = %loopentry
        ret i32 0
}

