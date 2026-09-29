; RUN: llc < %s

;; Date:     Jul 8, 2003.
;; From:     test/Programs/MultiSource/Olden-perimeter
;; Function: int %adj(uint %d.1, uint %ct.1)
;;
;; Errors: (1) cast-int-to-bool was being treated as a NOP (i.e., the int
;;	       register was treated as effectively true if non-zero).
;;	       This cannot be used for later boolean operations.
;;	   (2) (A or NOT(B)) was being folded into A orn B, which is ok
;;	       for bitwise operations but not booleans!  For booleans,
;;	       the result has to be compared with 0.

@.str_1 = internal constant [30 x i8] c"d = %d, ct = %d, d ^ ct = %d\0A\00"

declare i32 @printf(ptr, ...)

define i32 @adj(i32 %d.1, i32 %ct.1) {
entry:
        %tmp.19 = icmp eq i32 %ct.1, 2          ; <i1> [#uses=1]
        %tmp.22.not = trunc i32 %ct.1 to i1              ; <i1> [#uses=1]
        %tmp.221 = xor i1 %tmp.22.not, true             ; <i1> [#uses=1]
        %tmp.26 = or i1 %tmp.19, %tmp.221               ; <i1> [#uses=1]
        %tmp.27 = zext i1 %tmp.26 to i32                ; <i32> [#uses=1]
        ret i32 %tmp.27
}

define i32 @main() {
entry:
        %result = call i32 @adj( i32 3, i32 2 )         ; <i32> [#uses=1]
        %tmp.0 = call i32 (ptr, ...) @printf( ptr @.str_1, i32 3, i32 2, i32 %result )              ; <i32> [#uses=0]
        ret i32 0
}

