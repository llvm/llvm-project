; RUN: split-file %s %t
; RUN: not llvm-as < %t/non_integer_type.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=NON-INTEGER-TYPE
; RUN: not llvm-as < %t/missing_type.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING-TYPE
; RUN: not llvm-as < %t/empty.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMPTY
; RUN: not llvm-as < %t/bad_bounds.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=BAD-BOUNDS
; RUN: not llvm-as < %t/overlap.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=OVERLAP
; RUN: not llvm-as < %t/unordered.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNORDERED
; RUN: not llvm-as < %t/missing_comma.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=MISSING-COMMA

;--- non_integer_type.ll
; NON-INTEGER-TYPE: the rangeset must have integer type!
declare rangeset(float (0, 2)) float @non_integer_type()

;--- missing_type.ll
; MISSING-TYPE: expected type
declare rangeset((0, 2)) i32 @missing_type()

;--- empty.ll
; EMPTY: expected '('
declare rangeset(i32) i32 @empty()

;--- bad_bounds.ll
; BAD-BOUNDS: rangeset requires lower <= upper
declare rangeset(i32 (7, 5)) i32 @bad_bounds()

;--- overlap.ll
; OVERLAP: Invalid (unordered or overlapping) range set
declare rangeset(i32 (0, 2), (2, 4)) i32 @overlap()

;--- unordered.ll
; UNORDERED: Invalid (unordered or overlapping) range set
declare rangeset(i32 (5, 8), (0, 2)) i32 @unordered()

;--- missing_comma.ll
; MISSING-COMMA: expected ','
declare rangeset(i32 (0 2)) i32 @missing_comma()
