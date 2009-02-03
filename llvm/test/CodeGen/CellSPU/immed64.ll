; RUN: llvm-as -o - %s | llc -march=cellspu > %t1.s
; RUN: grep lqa        %t1.s | count 13
; RUN: grep ilhu       %t1.s | count 15
; RUN: grep ila        %t1.s | count 1
; RUN: grep -w il      %t1.s | count 6
; RUN: grep shufb      %t1.s | count 13
; RUN: grep      65520 %t1.s | count  1
; RUN: grep      43981 %t1.s | count  1
; RUN: grep      13702 %t1.s | count  1
; RUN: grep      28225 %t1.s | count  1
; RUN: grep      30720 %t1.s | count  1
; RUN: grep 3233857728 %t1.s | count  8
; RUN: grep 2155905152 %t1.s | count  6
; RUN: grep      66051 %t1.s | count  7
; RUN: grep  471670303 %t1.s | count 11

target datalayout = "E-p:32:32:128-f64:64:128-f32:32:128-i64:32:128-i32:32:128-i16:16:128-i8:8:128-i1:8:128-a0:0:128-v128:128:128-s0:128:128"
target triple = "spu"

;  1311768467750121234 => 0x 12345678 abcdef12 (4660,22136/43981,61202)
; 18446744073709551591 => 0x ffffffff ffffffe7 (-25)
; 18446744073708516742 => 0x ffffffff fff03586 (-1034874)
;              5308431 => 0x 00000000 0051000F
;  9223372038704560128 => 0x 80000000 6e417800

define i64 @i64_const_1() {
  ret i64  1311768467750121234          ;; Constant pool spill
}

define i64 @i64_const_2() {
  ret i64 18446744073709551591          ;; IL/SHUFB
}

define i64 @i64_const_3() {
  ret i64 18446744073708516742          ;; IHLU/IOHL/SHUFB
}

define i64 @i64_const_4() {
  ret i64              5308431          ;; ILHU/IOHL/SHUFB
}

define i64 @i64_const_5() {
  ret i64                  511          ;; IL/SHUFB
}

define i64 @i64_const_6() {
  ret i64                 -512          ;; IL/SHUFB
}

define i64 @i64_const_7() {
  ret i64  9223372038704560128          ;; IHLU/IOHL/SHUFB
}

define i64 @i64_const_8() {
  ret i64 0                             ;; IL
}

define i64 @i64_const_9() {
  ret i64 -1                            ;; IL
}

define i64 @i64_const_10() {
  ret i64 281470681808895                ;; IL 65535
}

; 0x4005bf0a8b145769 ->
;   (ILHU 0x4005 [16389]/IOHL 0xbf0a [48906])
;   (ILHU 0x8b14 [35604]/IOHL 0x5769 [22377])
define double @f64_const_1() {
 ret double 0x4005bf0a8b145769        ;; ILHU/IOHL via pattern
}
 
define double @f64_const_2() {
 ret double 0x0010000000000000
}

define double @f64_const_3() {
 ret double 0x7fefffffffffffff
}

define double @f64_const_4() {
 ret double 0x400921fb54442d18
}
 
define double @f64_const_5() {
  ret double 0xbff6a09e667f3bcd         ;; ILHU/IOHL via pattern
}
 
define double @f64_const_6() {
  ret double 0x3ff6a09e667f3bcd
}

define double @f64_const_7() {
  ret double 0.000000e+00
}
