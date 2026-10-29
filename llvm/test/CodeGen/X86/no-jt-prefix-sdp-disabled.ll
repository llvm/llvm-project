; RUN: llc %s -o - | FileCheck %s --check-prefix NO-PARTITION
; RUN: llc %s -partition-static-data-sections -o - | FileCheck %s --check-prefix PARTITION

; NO-PARTITION: .section        .rodata,"a",@progbits
; PARTITION:    .section        .rodata.unlikely.,"a",@progbits

target triple = "x86_64-grtev4-linux-gnu"

define fastcc { i64, i64 } @monument_pass_9_25_26() #0 {
  switch i32 poison, label %3 [
    i32 16, label %3
    i32 15, label %3
    i32 12, label %3
    i32 26, label %2
    i32 18, label %1
    i32 8, label %2
    i32 6, label %2
    i32 19, label %3
    i32 17, label %3
  ]

1:                                                ; preds = %0
  ret { i64, i64 } zeroinitializer

2:                                                ; preds = %0, %0, %0
  ret { i64, i64 } zeroinitializer

3:                                                ; preds = %0, %0, %0, %0, %0, %0
  ret { i64, i64 } zeroinitializer
}

attributes #0 = { cold }
