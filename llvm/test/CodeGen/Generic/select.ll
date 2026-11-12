; RUN: llc < %s

%Domain = type { ptr, i32, ptr, i32, i32, ptr, ptr }
@AConst = constant i32 123              ; <ptr> [#uses=1]

; Test setting values of different constants in registers.
; 
define void @testConsts(i32 %N, float %X) {
        %a = add i32 %N, 1              ; <i32> [#uses=0]
        %i = add i32 %N, 12345678               ; <i32> [#uses=0]
        %b = add i16 4, 3               ; <i16> [#uses=0]
        %c = fadd float %X, 0.000000e+00         ; <float> [#uses=0]
        %d = fadd float %X, 0x400921CAC0000000           ; <float> [#uses=0]
        %f = add i32 -1, 10             ; <i32> [#uses=0]
        %g = add i16 20, -1             ; <i16> [#uses=0]
        %j = add i16 -1, 30             ; <i16> [#uses=0]
        %h = add i8 40, -1              ; <i8> [#uses=0]
        %k = add i8 -1, 50              ; <i8> [#uses=0]
        ret void
}

; A SetCC whose result is used should produce instructions to
; compute the boolean value in a register.  One whose result
; is unused will only generate the condition code but not
; the boolean result.
; 
define void @unusedBool(ptr %x, ptr %y) {
        icmp eq ptr %x, %y             ; <i1>:1 [#uses=1]
        xor i1 %1, true         ; <i1>:2 [#uses=0]
        icmp ne ptr %x, %y             ; <i1>:3 [#uses=0]
        ret void
}

; A constant argument to a Phi produces a Cast instruction in the
; corresponding predecessor basic block.  This checks a few things:
; -- phi arguments coming from the bottom of the same basic block
;    (they should not be forward substituted in the machine code!)
; -- code generation for casts of various types
; -- use of immediate fields for integral constants of different sizes
; -- branch on a constant condition
; 
define void @mergeConstants(ptr %x, ptr %y) {
; <label>:0
        br label %Top

Top:            ; preds = %Next, %Top, %0
        phi i32 [ 0, %0 ], [ 1, %Top ], [ 524288, %Next ]               ; <i32>:1 [#uses=0]
        phi float [ 0.000000e+00, %0 ], [ 1.000000e+00, %Top ], [ 2.000000e+00, %Next ]         ; <float>:2 [#uses=0]
        phi double [ 5.000000e-01, %0 ], [ 1.500000e+00, %Top ], [ 2.500000e+00, %Next ]         
        phi i1 [ true, %0 ], [ false, %Top ], [ true, %Next ]           ; <i1>:4 [#uses=0]
        br i1 true, label %Top, label %Next

Next:           ; preds = %Top
        br label %Top
}



; A constant argument to a cast used only once should be forward substituted
; and loaded where needed, which happens is:
; -- User of cast has no immediate field
; -- User of cast has immediate field but constant is too large to fit
;    or constant is not resolved until later (e.g., global address)
; -- User of cast uses it as a call arg. or return value so it is an implicit
;    use but has to be loaded into a virtual register so that the reg.
;    allocator can allocate the appropriate phys. reg. for it
;  
define ptr @castconst(float) {
        %castbig = trunc i64 99999999 to i32            ; <i32> [#uses=1]
        %castsmall = trunc i64 1 to i32         ; <i32> [#uses=1]
        %usebig = add i32 %castbig, %castsmall          ; <i32> [#uses=0]
        %dummyl = load i64, ptr @AConst           ; <i64> [#uses=0]
        %castnull = inttoptr i64 0 to ptr              ; <ptr> [#uses=1]
        ret ptr %castnull
}

; Test branch-on-comparison-with-zero, in two ways:
; 1. can be folded
; 2. cannot be folded because result of comparison is used twice
;
define void @testbool(i32 %A, i32 %B) {
        br label %Top

Top:            ; preds = %loop, %0
        %D = add i32 %A, %B             ; <i32> [#uses=2]
        %E = sub i32 %D, -4             ; <i32> [#uses=1]
        %C = icmp sle i32 %E, 0         ; <i1> [#uses=1]
        br i1 %C, label %retlbl, label %loop

loop:           ; preds = %loop, %Top
        %F = add i32 %A, %B             ; <i32> [#uses=0]
        %G = sub i32 %D, -4             ; <i32> [#uses=1]
        %D.upgrd.1 = icmp sle i32 %G, 0         ; <i1> [#uses=1]
        %E.upgrd.2 = xor i1 %D.upgrd.1, true            ; <i1> [#uses=1]
        br i1 %E.upgrd.2, label %loop, label %Top

retlbl:         ; preds = %Top
        ret void
}


;; Test use of a boolean result in cast operations.
;; Requires converting a condition code result into a 0/1 value in a reg.
;; 
define i32 @castbool(i32 %A, i32 %B) {
bb0:
        %cond213 = icmp slt i32 %A, %B          ; <i1> [#uses=1]
        %cast110 = zext i1 %cond213 to i8               ; <i8> [#uses=1]
        %cast109 = zext i8 %cast110 to i32              ; <i32> [#uses=1]
        ret i32 %cast109
}

;; Test use of a boolean result in arithmetic and logical operations.
;; Requires converting a condition code result into a 0/1 value in a reg.
;; 
define i1 @boolexpr(i1 %b, i32 %N) {
        %b2 = icmp sge i32 %N, 0                ; <i1> [#uses=1]
        %b3 = and i1 %b, %b2            ; <i1> [#uses=1]
        ret i1 %b3
}

; Test branch on floating point comparison
;
define void @testfloatbool(float %x, float %y) {
        br label %Top

Top:            ; preds = %Top, %0
        %p = fadd float %x, %y           ; <float> [#uses=1]
        %z = fsub float %x, %y           ; <float> [#uses=1]
        %b = fcmp ole float %p, %z              ; <i1> [#uses=2]
        %c = xor i1 %b, true            ; <i1> [#uses=0]
        br i1 %b, label %Top, label %goon

goon:           ; preds = %Top
        ret void
}


; Test cases where an LLVM instruction requires no machine
; instructions (e.g., cast int* to long).  But there are 2 cases:
; 1. If the result register has only a single use and the use is in the
;    same basic block, the operand will be copy-propagated during
;    instruction selection.
; 2. If the result register has multiple uses or is in a different
;    basic block, it cannot (or will not) be copy propagated during
;    instruction selection.  It will generate a
;    copy instruction (add-with-0), but this copy should get coalesced
;    away by the register allocator.
;
define i32 @checkForward(i32 %N, ptr %A) {
bb2:
        %reg114 = shl i32 %N, 2         ; <i32> [#uses=1]
        %cast115 = sext i32 %reg114 to i64              ; <i64> [#uses=1]
        %cast116 = ptrtoint ptr %A to i64              ; <i64> [#uses=1]
        %reg116 = add i64 %cast116, %cast115            ; <i64> [#uses=1]
        %castPtr = inttoptr i64 %reg116 to ptr         ; <ptr> [#uses=1]
        %reg118 = load i32, ptr %castPtr            ; <i32> [#uses=1]
        %cast117 = sext i32 %reg118 to i64              ; <i64> [#uses=2]
        %reg159 = add i64 1234567, %cast117             ; <i64> [#uses=0]
        %reg160 = add i64 7654321, %cast117             ; <i64> [#uses=0]
        ret i32 0
}


; Test case for unary NOT operation constructed from XOR.
; 
define void @checkNot(i1 %b, i32 %i) {
        %notB = xor i1 %b, true         ; <i1> [#uses=1]
        %notI = xor i32 %i, -1          ; <i32> [#uses=2]
        %F = icmp sge i32 %notI, 100            ; <i1> [#uses=1]
        %J = add i32 %i, %i             ; <i32> [#uses=1]
        %andNotB = and i1 %F, %notB             ; <i1> [#uses=0]
        %andNotI = and i32 %J, %notI            ; <i32> [#uses=0]
        %notB2 = xor i1 true, %b                ; <i1> [#uses=0]
        %notI2 = xor i32 -1, %i         ; <i32> [#uses=0]
        ret void
}

; Test case for folding getelementptr into a load/store
;
define i32 @checkFoldGEP(ptr %D, i64 %idx) {
        %reg841 = getelementptr %Domain, ptr %D, i64 0, i32 1               ; <ptr> [#uses=1]
        %reg820 = load i32, ptr %reg841             ; <i32> [#uses=1]
        ret i32 %reg820
}

; Test case for scalarising a 1 element vselect
;
define <1 x i32> @checkScalariseVSELECT(<1 x i32> %a, <1 x i32> %b) {
        %cond = icmp uge <1 x i32> %a, %b
        %s = select <1 x i1> %cond, <1 x i32> %a, <1 x i32> %b
        ret <1 x i32> %s
}
