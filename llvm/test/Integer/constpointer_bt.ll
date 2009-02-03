; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; This testcase is primarily used for testing that global values can be used as 
; constant pointer initializers.  This is tricky because they can be forward
; declared and involves an icky bytecode encoding.  There is no meaningful 
; optimization that can be performed on this file, it is just here to test 
; assembly and disassembly.
;


@t3 = global i40 * @t1           ;; Forward reference
@t1 = global i40 4
@t4 = global i40 ** @t3		 ;; reference to reference

@t2 = global i40 * @t1

global float * @2                ;; Forward numeric reference
global float * @2                ;; Duplicate forward numeric reference
global float 0.0
global float * @2                ;; Numeric reference


@fptr = global void() * @f       ;; Forward ref method defn
declare void @"f"()               ;; External method

@sptr1   = global [11x i8]* @somestr		;; Forward ref to a constant
@somestr = constant [11x i8] c"hello world"
@sptr2   = global [11x i8]* @somestr


