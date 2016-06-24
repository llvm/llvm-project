; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

; Check for several conditions that should result in SSAT.
; For example, the base test is equivalent to
; x < -k ? -k : (x > k ? k : x) in C. All patterns that bound x
; to the interval [-k, k] where k is a power of 2 can be
; transformed into SSAT. At the end there are some tests
; checking that conditionals are not transformed if they don't
; match the right pattern.

;
; Base tests with different bit widths
;

; x < -k ? -k : (x > k ? k : x)
; 32-bit base test
define i32 @sat_base_32bit(i32 %x) #0 {
; CHECK-LABEL: sat_base_32bit:
; CHECK: ssat r0, #24, r0
entry:
  %cmpLow = icmp slt i32 %x, -8388608
  %cmpUp = icmp sgt i32 %x, 8388607
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %x
  %saturateLow = select i1 %cmpLow, i32 -8388608, i32 %saturateUp
  ret i32 %saturateLow
}

; x < -k ? -k : (x > k ? k : x)
; 16-bit base test
define i16 @sat_base_16bit(i16 %x) #0 {
; CHECK-LABEL: sat_base_16bit:
; CHECK: ssat r0, #12, r0
entry:
  %cmpLow = icmp slt i16 %x, -2048
  %cmpUp = icmp sgt i16 %x, 2047
  %saturateUp = select i1 %cmpUp, i16 2047, i16 %x
  %saturateLow = select i1 %cmpLow, i16 -2048, i16 %saturateUp
  ret i16 %saturateLow
}

; x < -k ? -k : (x > k ? k : x)
; 8-bit base test
define i8 @sat_base_8bit(i8 %x) #0 {
; CHECK-LABEL: sat_base_8bit:
; CHECK: ssat r0, #6, r0
entry:
  %cmpLow = icmp slt i8 %x, -32
  %cmpUp = icmp sgt i8 %x, 31
  %saturateUp = select i1 %cmpUp, i8 31, i8 %x
  %saturateLow = select i1 %cmpLow, i8 -32, i8 %saturateUp
  ret i8 %saturateLow
}

;
; Tests where the conditionals that check for upper and lower bounds,
; or the < and > operators, are arranged in different ways. Only some
; of the possible combinations that lead to SSAT are tested.
;

; x < -k ? -k : (x < k ? x : k)
define i32 @sat_lower_upper_1(i32 %x) #0 {
; CHECK-LABEL: sat_lower_upper_1:
; CHECK: ssat r0, #24, r0
entry:
  %cmpLow = icmp slt i32 %x, -8388608
  %cmpUp = icmp slt i32 %x, 8388607
  %saturateUp = select i1 %cmpUp, i32 %x, i32 8388607
  %saturateLow = select i1 %cmpLow, i32 -8388608, i32 %saturateUp
  ret i32 %saturateLow
}

; x > -k ? (x > k ? k : x) : -k
define i32 @sat_lower_upper_2(i32 %x) #0 {
; CHECK-LABEL: sat_lower_upper_2:
; CHECK: ssat    r0, #24, r0
entry:
  %cmpLow = icmp sgt i32 %x, -8388608
  %cmpUp = icmp sgt i32 %x, 8388607
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %x
  %saturateLow = select i1 %cmpLow, i32 %saturateUp, i32 -8388608
  ret i32 %saturateLow
}

; x < k ? (x < -k ? -k : x) : k
define i32 @sat_upper_lower_1(i32 %x) #0 {
; CHECK-LABEL: sat_upper_lower_1:
; CHECK: ssat    r0, #24, r0
entry:
  %cmpUp = icmp slt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, -8388608
  %saturateLow = select i1 %cmpLow, i32 -8388608, i32 %x
  %saturateUp = select i1 %cmpUp, i32 %saturateLow, i32 8388607
  ret i32 %saturateUp
}

; x > k ? k : (x < -k ? -k : x)
define i32 @sat_upper_lower_2(i32 %x) #0 {
; CHECK-LABEL: sat_upper_lower_2:
; CHECK: ssat    r0, #24, r0
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, -8388608
  %saturateLow = select i1 %cmpLow, i32 -8388608, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; k < x ? k : (x > -k ? x : -k)
define i32 @sat_upper_lower_3(i32 %x) #0 {
; CHECK-LABEL: sat_upper_lower_3:
; CHECK: ssat    r0, #24, r0
entry:
  %cmpUp = icmp slt i32 8388607, %x
  %cmpLow = icmp sgt i32 %x, -8388608
  %saturateLow = select i1 %cmpLow, i32 %x, i32 -8388608
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

;
; Miscellanea
;

; Check that >= and <= work the same as > and <
; k <= x ? k : (x >= -k ? x : -k)
define i32 @sat_le_ge(i32 %x) #0 {
; CHECK-LABEL: sat_le_ge:
; CHECK: ssat    r0, #24, r0
entry:
  %cmpUp = icmp sle i32 8388607, %x
  %cmpLow = icmp sge i32 %x, -8388608
  %saturateLow = select i1 %cmpLow, i32 %x, i32 -8388608
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

;
; The following tests check for patterns that should not transform
; into SSAT but are similar enough that could confuse the selector.
;

; x > k ? k : (x > -k ? -k : x)
; First condition upper-saturates, second doesn't lower-saturate.
define i32 @no_sat_missing_lower(i32 %x) #0 {
; CHECK-LABEL: no_sat_missing_lower
; CHECK-NOT: ssat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp sgt i32 %x, -8388608
  %saturateLow = select i1 %cmpLow, i32 -8388608, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; x < k ? k : (x < -k ? -k : x)
; Second condition lower-saturates, first doesn't upper-saturate.
define i32 @no_sat_missing_upper(i32 %x) #0 {
; CHECK-LABEL: no_sat_missing_upper:
; CHECK-NOT: ssat
entry:
  %cmpUp = icmp slt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, -8388608
  %saturateLow = select i1 %cmpLow, i32 -8388608, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; Lower constant is different in the select and in the compare
define i32 @no_sat_incorrect_constant(i32 %x) #0 {
; CHECK-LABEL: no_sat_incorrect_constant:
; CHECK-NOT: ssat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, -8388608
  %saturateLow = select i1 %cmpLow, i32 -8388607, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; The interval is not [k, ~k]
define i32 @no_sat_incorrect_interval(i32 %x) #0 {
; CHECK-LABEL: no_sat_incorrect_interval:
; CHECK-NOT: ssat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, -19088744
  %saturateLow = select i1 %cmpLow, i32 -19088744, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; The returned value (y) is not the same as the tested value (x).
define i32 @no_sat_incorrect_return(i32 %x, i32 %y) #0 {
; CHECK-LABEL: no_sat_incorrect_return:
; CHECK-NOT: ssat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, -8388608
  %saturateLow = select i1 %cmpLow, i32 -8388608, i32 %y
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; One of the values in a compare (y) is not the same as the rest
; of the compare and select values (x).
define i32 @no_sat_incorrect_compare(i32 %x, i32 %y) #0 {
; CHECK-LABEL: no_sat_incorrect_compare:
; CHECK-NOT: ssat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp slt i32 %y, -8388608
  %saturateLow = select i1 %cmpLow, i32 -8388608, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}
