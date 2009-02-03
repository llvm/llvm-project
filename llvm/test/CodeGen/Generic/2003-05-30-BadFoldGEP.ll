; RUN: llvm-as < %s | llc

;; Date:     May 28, 2003.
;; From:     test/Programs/External/SPEC/CINT2000/254.gap.llvm.bc
;; Function: int %OpenOutput(sbyte* %filename.1)
;;
;; Error:    A sequence of GEPs is folded incorrectly by llc during selection
;;	     causing an assertion about a dynamic casting error.
;;	     This code sequence was produced (correctly) by preselection
;;	     from a nested pair of ConstantExpr getelementptrs.
;;	     The code below is the output of preselection.
;;	     The original ConstantExprs are included in a comment.
;;
;; Cause:    FoldGetElemChain() was inserting an extra leading 0 even though
;;	     the first instruction in the sequence contributes no indices.
;;	     The next instruction contributes a leading non-zero so another
;;	     zero should not be added before it!
;;
        %FileType = type { i32, [256 x i8], i32, i32, i32, i32 }
@OutputFiles = external global [16 x %FileType]         ; <[16 x %FileType]*> [#uses=1]
@Output = internal global %FileType* null               ; <%FileType**> [#uses=1]

define internal i32 @OpenOutput(i8* %filename.1) {
entry:
        %tmp.0 = load %FileType** @Output               ; <%FileType*> [#uses=1]
        %tmp.4 = getelementptr %FileType* %tmp.0, i64 1         ; <%FileType*> [#uses=1]
        %addrOfGlobal = getelementptr [16 x %FileType]* @OutputFiles, i64 0             ; <[16 x %FileType]*> [#uses=1]
        %constantGEP = getelementptr [16 x %FileType]* %addrOfGlobal, i64 1             ; <[16 x %FileType]*> [#uses=1]
        %constantGEP.upgrd.1 = getelementptr [16 x %FileType]* %constantGEP, i64 0, i64 0               ; <%FileType*> [#uses=1]
        %tmp.10 = icmp eq %FileType* %tmp.4, %constantGEP.upgrd.1               ; <i1> [#uses=1]
        br i1 %tmp.10, label %return, label %endif.0

endif.0:                ; preds = %entry
        ret i32 0

return:         ; preds = %entry
        ret i32 1
}

