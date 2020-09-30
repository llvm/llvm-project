; RUN: opt -disable-output -S -passes=print-ir-similarity < %s 2>&1 | FileCheck --allow-empty %s

; Check to make sure that the IRSimilarityIdentifier and IRSimilarityPrinterPass
; return items only within the same function when there are different sets of
; instructions in functions.

; CHECK: 2 candidates of length 3.  Found in: 
; CHECK-NEXT:   Function: turtle,  Basic Block: (unnamed)
; CHECK-NEXT:   Function: turtle,  Basic Block: (unnamed)
; CHECK-NEXT: 2 candidates of length 5.  Found in: 
; CHECK-NEXT:   Function: fish,  Basic Block: entry
; CHECK-NEXT:   Function: fish,  Basic Block: entry

define linkonce_odr void @fish() {
entry:
  %0 = alloca i32, align 4
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 6, i32* %0, align 4
  store i32 1, i32* %1, align 4
  store i32 2, i32* %2, align 4
  store i32 3, i32* %3, align 4
  store i32 4, i32* %4, align 4
  store i32 5, i32* %5, align 4
  ret void
}

define void @turtle(i32* %0, i32* %1, i32* %2, i32* %3) {
  %a = load i32, i32* %0
  %b = load i32, i32* %1
  %c = load i32, i32* %2
  %d = load i32, i32* %3
  ret void
}
