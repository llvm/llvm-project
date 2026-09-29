; Test that global constructors are correctly ordered
;
; RUN: lli -jit-kind=orc %s | FileCheck %s
;
; CHECK: H1
; CHECK-NEXT: H2
; CHECK-NEXT: H3
; CHECK-NEXT: M1
; CHECK-NEXT: M2
; CHECK-NEXT: M3
; CHECK-NEXT: 1
; CHECK-NEXT: 2
; CHECK-NEXT: 3
; CHECK-NEXT: 4
; CHECK-NEXT: 5
; CHECK-NEXT: 6
; CHECK-NEXT: 7
; CHECK-NEXT: 8
; CHECK-NEXT: 9
; CHECK-NEXT: 10
; CHECK-NEXT: 11
; CHECK-NEXT: 12
; CHECK-NEXT: 13
; CHECK-NEXT: 14
; CHECK-NEXT: 15
; CHECK-NEXT: 16
; CHECK-NEXT: 17

declare i32 @puts(ptr)

@str.H1 = private constant [3 x i8] c"H1\00"
@str.H2 = private constant [3 x i8] c"H2\00"
@str.H3 = private constant [3 x i8] c"H3\00"
@str.M1 = private constant [3 x i8] c"M1\00"
@str.M2 = private constant [3 x i8] c"M2\00"
@str.M3 = private constant [3 x i8] c"M3\00"
@str.1 = private constant [2 x i8] c"1\00"
@str.2 = private constant [2 x i8] c"2\00"
@str.3 = private constant [2 x i8] c"3\00"
@str.4 = private constant [2 x i8] c"4\00"
@str.5 = private constant [2 x i8] c"5\00"
@str.6 = private constant [2 x i8] c"6\00"
@str.7 = private constant [2 x i8] c"7\00"
@str.8 = private constant [2 x i8] c"8\00"
@str.9 = private constant [2 x i8] c"9\00"
@str.10 = private constant [3 x i8] c"10\00"
@str.11 = private constant [3 x i8] c"11\00"
@str.12 = private constant [3 x i8] c"12\00"
@str.13 = private constant [3 x i8] c"13\00"
@str.14 = private constant [3 x i8] c"14\00"
@str.15 = private constant [3 x i8] c"15\00"
@str.16 = private constant [3 x i8] c"16\00"
@str.17 = private constant [3 x i8] c"17\00"
@llvm.global_ctors = appending global [23 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1024, ptr @medium.1, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.1, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.2, ptr null }, { i32, ptr, ptr } { i32 1, ptr @high.1, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.3, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.4, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.5, ptr null }, { i32, ptr, ptr } { i32 1, ptr @high.2, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.6, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.7, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.8, ptr null }, { i32, ptr, ptr } { i32 1024, ptr @medium.2, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.9, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.10, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.11, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.12, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.13, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.14, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.15, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.16, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @default.17, ptr null }, { i32, ptr, ptr } { i32 1024, ptr @medium.3, ptr null }, { i32, ptr, ptr } { i32 1, ptr @high.3, ptr null }]

define internal i32 @high.1() #0 {
  %call = tail call i32 @puts(ptr @str.H1)
  ret i32 0
}

define internal i32 @high.2() #0 {
  %call = tail call i32 @puts(ptr @str.H2)
  ret i32 0
}

define internal i32 @high.3() #0 {
  %call = tail call i32 @puts(ptr @str.H3)
  ret i32 0
}

define internal i32 @medium.1() #0 {
  %call = tail call i32 @puts(ptr @str.M1)
  ret i32 0
}

define internal i32 @medium.2() #0 {
  %call = tail call i32 @puts(ptr @str.M2)
  ret i32 0
}

define internal i32 @medium.3() #0 {
  %call = tail call i32 @puts(ptr @str.M3)
  ret i32 0
}

define internal i32 @default.1() #0 {
  %call = tail call i32 @puts(ptr @str.1)
  ret i32 0
}

define internal i32 @default.2() #0 {
  %call = tail call i32 @puts(ptr @str.2)
  ret i32 0
}

define internal i32 @default.3() #0 {
  %call = tail call i32 @puts(ptr @str.3)
  ret i32 0
}

define internal i32 @default.4() #0 {
  %call = tail call i32 @puts(ptr @str.4)
  ret i32 0
}

define internal i32 @default.5() #0 {
  %call = tail call i32 @puts(ptr @str.5)
  ret i32 0
}

define internal i32 @default.6() #0 {
  %call = tail call i32 @puts(ptr @str.6)
  ret i32 0
}

define internal i32 @default.7() #0 {
  %call = tail call i32 @puts(ptr @str.7)
  ret i32 0
}

define internal i32 @default.8() #0 {
  %call = tail call i32 @puts(ptr @str.8)
  ret i32 0
}

define internal i32 @default.9() #0 {
  %call = tail call i32 @puts(ptr @str.9)
  ret i32 0
}

define internal i32 @default.10() #0 {
  %call = tail call i32 @puts(ptr @str.10)
  ret i32 0
}

define internal i32 @default.11() #0 {
  %call = tail call i32 @puts(ptr @str.11)
  ret i32 0
}

define internal i32 @default.12() #0 {
  %call = tail call i32 @puts(ptr @str.12)
  ret i32 0
}

define internal i32 @default.13() #0 {
  %call = tail call i32 @puts(ptr @str.13)
  ret i32 0
}

define internal i32 @default.14() #0 {
  %call = tail call i32 @puts(ptr @str.14)
  ret i32 0
}

define internal i32 @default.15() #0 {
  %call = tail call i32 @puts(ptr @str.15)
  ret i32 0
}

define internal i32 @default.16() #0 {
  %call = tail call i32 @puts(ptr @str.16)
  ret i32 0
}

define internal i32 @default.17() #0 {
  %call = tail call i32 @puts(ptr @str.17)
  ret i32 0
}

define i32 @main() {
  ret i32 0
}
