; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


; This testcase is for testing expressions constructed from
; constant values, including constant pointers to globals.
;

;;-------------------------------
;; Test constant cast expressions
;;-------------------------------

global i63 u0x00001     ; hexadecimal unsigned integer constants
global i63  s0x012312   ; hexadecimal signed integer constants

@t2 = global i33* @t1                             ;; Forward reference without cast
@t3 = global i33* bitcast (i33* @t1 to i33*)       ;; Forward reference with cast
@t1 = global i33 4                                ;; i32* @0
@t4 = global i33** bitcast (i33** @t3 to i33**)     ;; Cast of a previous cast
@t5 = global i33** @t3                           ;; Reference to a previous cast
@t6 = global i33*** @t4
@t7 = global float* inttoptr (i32 12345678 to float*) ;; Cast ordinary value to ptr
@t9 = global i33 fptosi (float sitofp (i33 8 to float) to i33) ;; Nested cast expression


global i32* bitcast (float* @4 to i32*)   ;; Forward numeric reference
global float* @4                       ;; Duplicate forward numeric reference
global float 0.0


;;---------------------------------------------------
;; Test constant getelementpr expressions for arrays
;;---------------------------------------------------

@array  = constant [2 x i33] [ i33 12, i33 52 ]
@arrayPtr = global i33* getelementptr ([2 x i33]* @array, i64 0, i64 0)    ;; i33* &@array[0][0]
@arrayPtr5 = global i33** getelementptr (i33** @arrayPtr, i64 5)    ;; i33* &@arrayPtr[5]

@somestr = constant [11x i8] c"hello world"
@char5  = global i8* getelementptr([11x i8]* @somestr, i64 0, i64 5)

;; cast of getelementptr
@char8a = global i33* bitcast (i8* getelementptr([11x i8]* @somestr, i64 0, i64 8) to i33*)

;; getelementptr containing casts
@char8b = global i8* getelementptr([11x i8]* @somestr, i64 sext (i8 0 to i64), i64 sext (i8 8 to i64))

;;-------------------------------------------------------
;; TODO: Test constant getelementpr expressions for structures
;;-------------------------------------------------------

%SType  = type { i33 , {float, {i8} }, i64 } ;; struct containing struct
%SAType = type { i33 , {[2x float], i64} } ;; struct containing array

@S1 = global %SType* null			;; Global initialized to NULL
@S2c = constant %SType { i33 1, {float,{i8}} {float 2.0, {i8} {i8 3}}, i64 4}

@S3c = constant %SAType { i33 1, {[2x float], i64} {[2x float] [float 2.0, float 3.0], i64 4} }

@S1ptr = global %SType** @S1		    ;; Ref. to global S1
@S2  = global %SType* @S2c		    ;; Ref. to constant S2
@S3  = global %SAType* @S3c		    ;; Ref. to constant S3

					    ;; Pointer to float (**@S1).1.0
@S1fld1a = global float* getelementptr (%SType* @S2c, i64 0, i32 1, i32 0)
					    ;; Another ptr to the same!
@S1fld1b = global float* getelementptr (%SType* @S2c, i64 0, i32 1, i32 0)

@S1fld1bptr = global float** @S1fld1b	    ;; Ref. to previous pointer

					    ;; Pointer to i8 (**@S2).1.1.0
@S2fld3 = global i8* getelementptr (%SType* @S2c, i64 0, i32 1, i32 1, i32 0) 

					    ;; Pointer to float (**@S2).1.0[0]
;@S3fld3 = global float* getelementptr (%SAType** @S3, i64 0, i64 0, i32 1, i32 0, i64 0) 

;;---------------------------------------------------------
;; TODO: Test constant expressions for unary and binary operators
;;---------------------------------------------------------

;;---------------------------------------------------


