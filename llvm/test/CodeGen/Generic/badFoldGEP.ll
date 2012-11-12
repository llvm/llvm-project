; RUN: llc < %s

;; GetMemInstArgs() folded the two getElementPtr instructions together,
;; producing an illegal getElementPtr.  That's because the type generated
;; by the last index for the first one is a structure field, not an array
;; element, and the second one indexes off that structure field.
;; The code is legal but not type-safe and the two GEPs should not be folded.
;; 
;; This code fragment is from Spec/CINT2000/197.parser/197.parser.bc,
;; file post_process.c, function build_domain().
;; (Modified to replace store with load and return load value.)
;; 
        %Domain = type { i8*, i32, i32*, i32, i32, i32*, %Domain* }
@domain_array = external global [497 x %Domain]         ; <[497 x %Domain]*> [#uses=2]

declare void @opaque([497 x %Domain]*)

define i32 @main(i32 %argc, i8** %argv) {
bb0:
        call void @opaque( [497 x %Domain]* @domain_array )
        %cann-indvar-idxcast = sext i32 %argc to i64            ; <i64> [#uses=1]
        %reg841 = getelementptr [497 x %Domain]* @domain_array, i64 0, i64 %cann-indvar-idxcast, i32 3          ; <i32*> [#uses=1]
        %reg846 = getelementptr i32* %reg841, i64 1             ; <i32*> [#uses=1]
        %reg820 = load i32* %reg846             ; <i32> [#uses=1]
        ret i32 %reg820
}

