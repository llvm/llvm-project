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
        %Domain = type { ptr, i32, ptr, i32, i32, ptr, ptr }
@domain_array = external global [497 x %Domain]         ; <ptr> [#uses=2]

declare void @opaque(ptr)

define i32 @main(i32 %argc, ptr %argv) {
bb0:
        call void @opaque( ptr @domain_array )
        %cann-indvar-idxcast = sext i32 %argc to i64            ; <i64> [#uses=1]
        %reg841 = getelementptr [497 x %Domain], ptr @domain_array, i64 0, i64 %cann-indvar-idxcast, i32 3          ; <ptr> [#uses=1]
        %reg846 = getelementptr i32, ptr %reg841, i64 1             ; <ptr> [#uses=1]
        %reg820 = load i32, ptr %reg846             ; <i32> [#uses=1]
        ret i32 %reg820
}

