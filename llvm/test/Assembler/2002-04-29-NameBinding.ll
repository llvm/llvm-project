; There should be NO references to the global v1.  The local v1 should
; have all of the references!
;
; Check by running globaldce, which will remove the constant if there are
; no references to it!
; 
; RUN: llvm-as < %s | opt -globaldce | llvm-dis | \
; RUN:   not grep constant
;

@v1 = internal constant i32 5           

define i32 @createtask() {
        %v1 = alloca i32                ;; Alloca should have one use! 
        %reg112 = load i32* %v1         ;; This load should not use the global!
        ret i32 %reg112
}

