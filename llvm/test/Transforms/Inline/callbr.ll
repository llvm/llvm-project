; RUN: opt -inline -S < %s | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -S < %s | FileCheck %s
; RUN: opt -passes='module-inline' -S < %s | FileCheck %s

define dso_local i32 @main() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = call i32 @t32(i32 0)
  ret i32 %2
}

define internal i32 @t32(i32) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  callbr void asm sideeffect "testl $0, $0; jne ${1:l};", "r,X,X,~{dirflag},~{fpsr},~{flags}"(i32 %4, i8* blockaddress(@t32, %7), i8* blockaddress(@t32, %6)) #1
          to label %5 [label %7, label %6]

; <label>:5:                                      ; preds = %1
  store i32 0, i32* %2, align 4
  br label %8

; <label>:6:                                      ; preds = %1
  store i32 1, i32* %2, align 4
  br label %8

; <label>:7:                                      ; preds = %1
  store i32 2, i32* %2, align 4
  br label %8

; <label>:8:                                      ; preds = %7, %6, %5
  %9 = load i32, i32* %2, align 4
  ret i32 %9
}

; Check that @t32 no longer exists after inlining, as it has now been inlined
; into @main.

; CHECK-NOT: @t32
; CHECK: define dso_local i32 @main
; CHECK: callbr void asm sideeffect "testl $0, $0; jne ${1:l};", "r,X,X,~{dirflag},~{fpsr},~{flags}"(i32 %6, i8* blockaddress(@main, %9), i8* blockaddress(@main, %8))
; CHECK: to label %7 [label %9, label %8]
; CHECK: 7:
; CHECK-NEXT: store i32 0, i32* %1, align 4
; CHECK-NEXT: br label %t32.exit
; CHECK: 8:
; CHECK-NEXT: store i32 1, i32* %1, align 4
; CHECK-NEXT: br label %t32.exit
; CHECK: 9:
; CHECK-NEXT: store i32 2, i32* %1, align 4
; CHECK-NEXT: br label %t32.exit
; CHECK: t32.exit:
; CHECK-NEXT: %10 = load i32, i32* %1, align 4
; CHECK: ret i32 %10
