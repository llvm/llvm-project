; RUN: llc < %s -mtriple=x86_64-pc-linux -group-functions-by-hotness=true | FileCheck %s -check-prefix=PARTITION
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -group-functions-by-hotness=false | FileCheck %s -check-prefix=NO-PARTITION-FUNCTION-SECTION
; RUN: llc < %s -mtriple=x86_64-pc-linux -function-sections -group-functions-by-hotness=true | FileCheck %s -check-prefix=PARTITION-FUNCTION-SECTION

; PARTITION: .text.unlikely
; PARTITION: .globl  _Z3foov
; NO-PARTITION-FUNCTION-SECTION: .text._Z3foov
; PARTITION-FUNCTION-SECTION: .text.unlikely._Z3foov
define i32 @_Z3foov() #0 {
  ret i32 0
}

; PARTITION: .globl  _Z3barv
; NO-PARTITION-FUNCTION-SECTION: .text._Z3barv
; PARTITION-FUNCTION-SECTION: .text.unlikely._Z3barv
define i32 @_Z3barv() #1 !prof !0 {
  ret i32 1
}

; PARTITION: .text
; PARTITION: .globl  _Z3bazv
; NO-PARTITION-FUNCTION-SECTION: .text._Z3bazv
; PARTITION-FUNCTION-SECTION: .text._Z3bazv
define i32 @_Z3bazv() #1 {
  ret i32 2
}

attributes #0 = { nounwind uwtable cold }
attributes #1 = { nounwind uwtable }

!0 = !{!"function_entry_count", i64 0}
