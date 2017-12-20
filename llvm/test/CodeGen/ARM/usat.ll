; RUN: llc -mtriple=armv4t-eabi %s -o -  | FileCheck %s --check-prefix=CHECK --check-prefix=V4T
; RUN: llc -mtriple=armv6-eabi %s -o -   | FileCheck %s --check-prefix=CHECK --check-prefix=V6
; RUN: llc -mtriple=armv6t2-eabi %s -o - | FileCheck %s --check-prefix=CHECK --check-prefix=V6T2

; Check for several conditions that should result in USAT.
; For example, the base test is equivalent to
; x < 0 ? 0 : (x > k ? k : x) in C. All patterns that bound x
; to the interval [0, k] where k + 1 is a power of 2 can be
; transformed into USAT. At the end there are some tests
; checking that conditionals are not transformed if they don't
; match the right pattern.

;
; Base tests with different bit widths
;

; x < 0 ? 0 : (x > k ? k : x)
; 32-bit base test
define i32 @unsigned_sat_base_32bit(i32 %x) #0 {
; CHECK-LABEL: unsigned_sat_base_32bit:
; V6: usat r0, #23, r0
; V6T2: usat r0, #23, r0
; V4T-NOT: usat
entry:
  %cmpLow = icmp slt i32 %x, 0
  %cmpUp = icmp sgt i32 %x, 8388607
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %x
  %saturateLow = select i1 %cmpLow, i32 0, i32 %saturateUp
  ret i32 %saturateLow
}

; x < 0 ? 0 : (x > k ? k : x)
; 16-bit base test
define i16 @unsigned_sat_base_16bit(i16 %x) #0 {
; CHECK-LABEL: unsigned_sat_base_16bit:
; V6: usat r0, #11, r0
; V6T2: usat r0, #11, r0
; V4T-NOT: usat
entry:
  %cmpLow = icmp slt i16 %x, 0
  %cmpUp = icmp sgt i16 %x, 2047
  %saturateUp = select i1 %cmpUp, i16 2047, i16 %x
  %saturateLow = select i1 %cmpLow, i16 0, i16 %saturateUp
  ret i16 %saturateLow
}

; x < 0 ? 0 : (x > k ? k : x)
; 8-bit base test
define i8 @unsigned_sat_base_8bit(i8 %x) #0 {
; CHECK-LABEL: unsigned_sat_base_8bit:
; V6: usat r0, #5, r0
; V6T2: usat r0, #5, r0
; V4T-NOT: usat
entry:
  %cmpLow = icmp slt i8 %x, 0
  %cmpUp = icmp sgt i8 %x, 31
  %saturateUp = select i1 %cmpUp, i8 31, i8 %x
  %saturateLow = select i1 %cmpLow, i8 0, i8 %saturateUp
  ret i8 %saturateLow
}

;
; Tests where the conditionals that check for upper and lower bounds,
; or the < and > operators, are arranged in different ways. Only some
; of the possible combinations that lead to USAT are tested.
;
; x < 0 ? 0 : (x < k ? x : k)
define i32 @unsigned_sat_lower_upper_1(i32 %x) #0 {
; CHECK-LABEL: unsigned_sat_lower_upper_1:
; V6: usat r0, #23, r0
; V6T2: usat r0, #23, r0
; V4T-NOT: usat
entry:
  %cmpLow = icmp slt i32 %x, 0
  %cmpUp = icmp slt i32 %x, 8388607
  %saturateUp = select i1 %cmpUp, i32 %x, i32 8388607
  %saturateLow = select i1 %cmpLow, i32 0, i32 %saturateUp
  ret i32 %saturateLow
}

; x > 0 ? (x > k ? k : x) : 0
define i32 @unsigned_sat_lower_upper_2(i32 %x) #0 {
; CHECK-LABEL: unsigned_sat_lower_upper_2:
; V6: usat    r0, #23, r0
; V6T2: usat    r0, #23, r0
; V4T-NOT: usat
entry:
  %cmpLow = icmp sgt i32 %x, 0
  %cmpUp = icmp sgt i32 %x, 8388607
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %x
  %saturateLow = select i1 %cmpLow, i32 %saturateUp, i32 0
  ret i32 %saturateLow
}

; x < k ? (x < 0 ? 0 : x) : k
define i32 @unsigned_sat_upper_lower_1(i32 %x) #0 {
; CHECK-LABEL: unsigned_sat_upper_lower_1:
; V6: usat    r0, #23, r0
; V6T2: usat    r0, #23, r0
; V4T-NOT: usat
entry:
  %cmpUp = icmp slt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, 0
  %saturateLow = select i1 %cmpLow, i32 0, i32 %x
  %saturateUp = select i1 %cmpUp, i32 %saturateLow, i32 8388607
  ret i32 %saturateUp
}

; x > k ? k : (x < 0 ? 0 : x)
define i32 @unsigned_sat_upper_lower_2(i32 %x) #0 {
; CHECK-LABEL: unsigned_sat_upper_lower_2:
; V6: usat    r0, #23, r0
; V6T2: usat    r0, #23, r0
; V4T-NOT: usat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, 0
  %saturateLow = select i1 %cmpLow, i32 0, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; k < x ? k : (x > 0 ? x : 0)
define i32 @unsigned_sat_upper_lower_3(i32 %x) #0 {
; CHECK-LABEL: unsigned_sat_upper_lower_3:
; V6: usat    r0, #23, r0
; V6T2: usat    r0, #23, r0
; V4T-NOT: usat
entry:
  %cmpUp = icmp slt i32 8388607, %x
  %cmpLow = icmp sgt i32 %x, 0
  %saturateLow = select i1 %cmpLow, i32 %x, i32 0
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

;
; The following tests check for patterns that should not transform
; into USAT but are similar enough that could confuse the selector.
;
; x > k ? k : (x > 0 ? 0 : x)
; First condition upper-saturates, second doesn't lower-saturate.
define i32 @no_unsigned_sat_missing_lower(i32 %x) #0 {
; CHECK-LABEL: no_unsigned_sat_missing_lower
; CHECK-NOT: usat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp sgt i32 %x, 0
  %saturateLow = select i1 %cmpLow, i32 0, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; x < k ? k : (x < 0 ? 0 : x)
; Second condition lower-saturates, first doesn't upper-saturate.
define i32 @no_unsigned_sat_missing_upper(i32 %x) #0 {
; CHECK-LABEL: no_unsigned_sat_missing_upper:
; CHECK-NOT: usat
entry:
  %cmpUp = icmp slt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, 0
  %saturateLow = select i1 %cmpLow, i32 0, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; Lower constant is different in the select and in the compare
define i32 @no_unsigned_sat_incorrect_constant(i32 %x) #0 {
; CHECK-LABEL: no_unsigned_sat_incorrect_constant:
; CHECK-NOT: usat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, 0
  %saturateLow = select i1 %cmpLow, i32 -1, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; The interval is not [0, k]
define i32 @no_unsigned_sat_incorrect_interval(i32 %x) #0 {
; CHECK-LABEL: no_unsigned_sat_incorrect_interval:
; CHECK-NOT: usat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, -4
  %saturateLow = select i1 %cmpLow, i32 -4, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; The returned value (y) is not the same as the tested value (x).
define i32 @no_unsigned_sat_incorrect_return(i32 %x, i32 %y) #0 {
; CHECK-LABEL: no_unsigned_sat_incorrect_return:
; CHECK-NOT: usat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp slt i32 %x, 0
  %saturateLow = select i1 %cmpLow, i32 0, i32 %y
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}

; One of the values in a compare (y) is not the same as the rest
; of the compare and select values (x).
define i32 @no_unsigned_sat_incorrect_compare(i32 %x, i32 %y) #0 {
; CHECK-LABEL: no_unsigned_sat_incorrect_compare:
; CHECK-NOT: usat
entry:
  %cmpUp = icmp sgt i32 %x, 8388607
  %cmpLow = icmp slt i32 %y, 0
  %saturateLow = select i1 %cmpLow, i32 0, i32 %x
  %saturateUp = select i1 %cmpUp, i32 8388607, i32 %saturateLow
  ret i32 %saturateUp
}
