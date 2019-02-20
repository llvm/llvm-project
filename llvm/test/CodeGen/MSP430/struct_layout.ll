; RUN: llc < %s | FileCheck %s

target triple = "msp430"

%struct.X = type { i8 }

; CHECK-LABEL: @foo
; CHECK: sub   #4, r1
; CHECK: mov.b   #1, 3(r1)
define void @foo() {
  %1 = alloca %struct.X
  %2 = alloca %struct.X
  %3 = alloca %struct.X
  %4 = getelementptr inbounds %struct.X, %struct.X* %1, i32 0, i32 0
  store i8 1, i8* %4
  %5 = getelementptr inbounds %struct.X, %struct.X* %2, i32 0, i32 0
  store i8 1, i8* %5
  %6 = getelementptr inbounds %struct.X, %struct.X* %3, i32 0, i32 0
  store i8 1, i8* %6
  ret void
}

; CHECK-LABEL: @bar
; CHECK: sub   #4, r1
; CHECK: mov.b   #1, 3(r1)
define void @bar() {
  %1 = alloca [3 x %struct.X]
  %2 = getelementptr inbounds [3 x %struct.X], [3 x %struct.X]* %1, i16 0, i16 0
  %3 = getelementptr inbounds %struct.X, %struct.X* %2, i32 0, i32 0
  store i8 1, i8* %3
  %4 = getelementptr inbounds [3 x %struct.X], [3 x %struct.X]* %1, i16 0, i16 1
  %5 = getelementptr inbounds %struct.X, %struct.X* %4, i32 0, i32 0
  store i8 1, i8* %5
  %6 = getelementptr inbounds [3 x %struct.X], [3 x %struct.X]* %1, i16 0, i16 2
  %7 = getelementptr inbounds %struct.X, %struct.X* %6, i32 0, i32 0
  store i8 1, i8* %7
  ret void
}

%struct.Y = type { i8, i16 }

; CHECK-LABEL: @baz
; CHECK: sub   #8, r1
; CHECK: mov   #2, 6(r1)
define void @baz() {
  %1 = alloca %struct.Y, align 2
  %2 = alloca %struct.Y, align 2
  %3 = getelementptr inbounds %struct.Y, %struct.Y* %1, i32 0, i32 0
  store i8 1, i8* %3, align 2
  %4 = getelementptr inbounds %struct.Y, %struct.Y* %1, i32 0, i32 1
  store i16 2, i16* %4, align 2
  %5 = getelementptr inbounds %struct.Y, %struct.Y* %2, i32 0, i32 0
  store i8 1, i8* %5, align 2
  %6 = getelementptr inbounds %struct.Y, %struct.Y* %2, i32 0, i32 1
  store i16 2, i16* %6, align 2
  ret void
}
