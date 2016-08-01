; RUN: opt < %s -simplifycfg -S | FileCheck %s

target datalayout = "e-n32"

; CHECK-LABEL: @test1
; CHECK: %1 = sub i32 %a, 97
; CHECK-DAG: %2 = lshr i32 %1, 2
; CHECK-DAG: %3 = shl i32 %1, 30
; CHECK: %4 = or i32 %2, %3
; CHECK:  switch i32 %4, label %def [
; CHECK:    i32 0, label %one
; CHECK:    i32 1, label %two
; CHECK:    i32 2, label %three
; CHECK:  ]
define i32 @test1(i32 %a) {
  switch i32 %a, label %def [
    i32 97, label %one
    i32 101, label %two
    i32 105, label %three
    i32 109, label %three
  ]

def:
  ret i32 8867

one:
  ret i32 11984
two:
  ret i32 1143
three:
  ret i32 99783
}

; Optimization shouldn't trigger; bitwidth > 64
; CHECK-LABEL: @test2
; CHECK: switch i128 %a, label %def
define i128 @test2(i128 %a) {
  switch i128 %a, label %def [
    i128 97, label %one
    i128 101, label %two
    i128 105, label %three
    i128 109, label %three
  ]

def:
  ret i128 8867

one:
  ret i128 11984
two:
  ret i128 1143
three:
  ret i128 99783
}


; Optimization shouldn't trigger; no holes present
; CHECK-LABEL: @test3
; CHECK: switch i32 %a, label %def
define i32 @test3(i32 %a) {
  switch i32 %a, label %def [
    i32 97, label %one
    i32 98, label %two
    i32 99, label %three
  ]

def:
  ret i32 8867

one:
  ret i32 11984
two:
  ret i32 1143
three:
  ret i32 99783
}

; Optimization shouldn't trigger; not an arithmetic progression
; CHECK-LABEL: @test4
; CHECK: switch i32 %a, label %def
define i32 @test4(i32 %a) {
  switch i32 %a, label %def [
    i32 97, label %one
    i32 102, label %two
    i32 105, label %three
    i32 109, label %three
  ]

def:
  ret i32 8867

one:
  ret i32 11984
two:
  ret i32 1143
three:
  ret i32 99783
}

; Optimization shouldn't trigger; not a power of two
; CHECK-LABEL: @test5
; CHECK: switch i32 %a, label %def
define i32 @test5(i32 %a) {
  switch i32 %a, label %def [
    i32 97, label %one
    i32 102, label %two
    i32 107, label %three
    i32 112, label %three
  ]

def:
  ret i32 8867

one:
  ret i32 11984
two:
  ret i32 1143
three:
  ret i32 99783
}

; CHECK-LABEL: @test6
; CHECK: %1 = sub i32 %a, -109
; CHECK-DAG: %2 = lshr i32 %1, 2
; CHECK-DAG: %3 = shl i32 %1, 30
; CHECK: %4 = or i32 %2, %3
; CHECK:  switch i32 %4, label %def [
define i32 @test6(i32 %a) optsize {
  switch i32 %a, label %def [
    i32 -97, label %one
    i32 -101, label %two
    i32 -105, label %three
    i32 -109, label %three
  ]

def:
  ret i32 8867

one:
  ret i32 11984
two:
  ret i32 1143
three:
  ret i32 99783
}

; CHECK-LABEL: @test7
; CHECK: %1 = sub i8 %a, -36
; CHECK-DAG: %2 = lshr i8 %1, 2
; CHECK-DAG: %3 = shl i8 %1, 6
; CHECK: %4 = or i8 %2, %3
; CHECK:  switch.tableidx = {{.*}} %4
define i8 @test7(i8 %a) optsize {
  switch i8 %a, label %def [
    i8 220, label %one
    i8 224, label %two
    i8 228, label %three
    i8 232, label %three
  ]

def:
  ret i8 8867

one:
  ret i8 11984
two:
  ret i8 1143
three:
  ret i8 99783
}

; CHECK-LABEL: @test8
; CHECK: %1 = sub i32 %a, 97
; CHECK-DAG: %2 = lshr i32 %1, 2
; CHECK-DAG: %3 = shl i32 %1, 30
; CHECK: %4 = or i32 %2, %3
; CHECK:  switch i32 %4, label %def [
define i32 @test8(i32 %a) optsize {
  switch i32 %a, label %def [
    i32 97, label %one
    i32 101, label %two
    i32 105, label %three
    i32 113, label %three
  ]

def:
  ret i32 8867

one:
  ret i32 11984
two:
  ret i32 1143
three:
  ret i32 99783
}

; CHECK-LABEL: @test9
; CHECK:  switch
; CHECK:  i32 6
; CHECK:  i32 7
; CHECK:  i32 0
; CHECK:  i32 2
define i32 @test9(i32 %a) {
  switch i32 %a, label %def [
    i32 18, label %one
    i32 20, label %two
    i32 6, label %three
    i32 10, label %three
  ]

def:
  ret i32 8867

one:
  ret i32 11984
two:
  ret i32 1143
three:
  ret i32 99783
}

