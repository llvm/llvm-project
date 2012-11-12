; RUN: llc -mtriple=i386-apple-macosx < %s | FileCheck %s
; rdar://12396696

@JT = global [4 x i32] [i32 sub (i32 ptrtoint (i8* blockaddress(@h, %18) to i32), i32 ptrtoint (i8* blockaddress(@h, %11) to i32)), i32 sub (i32 ptrtoint (i8* blockaddress(@h, %17) to i32), i32 ptrtoint (i8* blockaddress(@h, %11) to i32)), i32 sub (i32 ptrtoint (i8* blockaddress(@h, %22) to i32), i32 ptrtoint (i8* blockaddress(@h, %18) to i32)), i32 sub (i32 ptrtoint (i8* blockaddress(@h, %22) to i32), i32 ptrtoint (i8* blockaddress(@h, %17) to i32))]
@gGlobalLock = external global i8*
@.str40 = external global [35 x i8]

; CHECK: _JT:
; CHECK-NOT: .long Ltmp{{[0-9]+}}-1
; CHECK-NOT: .long 1-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}

define void @h(i8*) nounwind ssp {
  %2 = alloca i8*
  store i8* %0, i8** %2
  %3 = load i8** %2
  %4 = bitcast i8* %3 to { i32, i32 }*
  %5 = getelementptr { i32, i32 }* %4, i32 0, i32 0
  %6 = load i32* %5
  %7 = srem i32 %6, 2
  %8 = icmp slt i32 %6, 2
  %9 = select i1 %8, i32 %6, i32 %7
  %10 = icmp eq i32 %9, 0
  br label %11

; <label>:11                                      ; preds = %1
  %12 = zext i1 %10 to i32
  %13 = getelementptr [4 x i32]* @JT, i32 0, i32 %12
  %14 = load i32* %13
  %15 = add i32 %14, ptrtoint (i8* blockaddress(@h, %11) to i32)
  %16 = inttoptr i32 %15 to i8*
  indirectbr i8* %16, [label %17, label %18]

; <label>:17                                      ; preds = %11
  tail call void (i8*, ...)* @g(i8* getelementptr inbounds ([35 x i8]* @.str40, i32 0, i32 0))
  br label %22

; <label>:18                                      ; preds = %11
  %19 = call i32 @f(i32 -1037694186) nounwind
  %20 = inttoptr i32 %19 to i32 (i8**)*
  %21 = tail call i32 %20(i8** @gGlobalLock)
  br label %22

; <label>:22                                      ; preds = %18, %17
  ret void
}

declare i32 @f(i32)

declare void @g(i8*, ...)
