; RUN: llc -mtriple=i386-apple-macosx < %s | FileCheck %s
; rdar://12396696

@JT = global [4 x i32] [i32 sub (i32 ptrtoint (i8* blockaddress(@h, %bb16) to i32), i32 ptrtoint (i8* blockaddress(@h, %bb9) to i32)), i32 sub (i32 ptrtoint (i8* blockaddress(@h, %bb15) to i32), i32 ptrtoint (i8* blockaddress(@h, %bb9) to i32)), i32 sub (i32 ptrtoint (i8* blockaddress(@h, %bb20) to i32), i32 ptrtoint (i8* blockaddress(@h, %bb16) to i32)), i32 sub (i32 ptrtoint (i8* blockaddress(@h, %bb20) to i32), i32 ptrtoint (i8* blockaddress(@h, %bb15) to i32))]
@gGlobalLock = external global i8*
@.str40 = external global [35 x i8]

; CHECK: _JT:
; CHECK-NOT: .long Ltmp{{[0-9]+}}-1
; CHECK-NOT: .long 1-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}

define void @h(i8* %arg) nounwind ssp {
bb:
  %i = alloca i8*, align 8
  store i8* %arg, i8** %i, align 8
  %i1 = load i8*, i8** %i, align 8
  %i2 = bitcast i8* %i1 to { i32, i32 }*
  %i3 = getelementptr { i32, i32 }, { i32, i32 }* %i2, i32 0, i32 0
  %i4 = load i32, i32* %i3, align 4
  %i5 = srem i32 %i4, 2
  %i6 = icmp slt i32 %i4, 2
  %i7 = select i1 %i6, i32 %i4, i32 %i5
  %i8 = icmp eq i32 %i7, 0
  br label %bb9

bb9:                                              ; preds = %bb
  %i10 = zext i1 %i8 to i32
  %i11 = getelementptr [4 x i32], [4 x i32]* @JT, i32 0, i32 %i10
  %i12 = load i32, i32* %i11, align 4
  %i13 = add i32 %i12, ptrtoint (i8* blockaddress(@h, %bb9) to i32)
  %i14 = inttoptr i32 %i13 to i8*
  indirectbr i8* %i14, [label %bb15, label %bb16]

bb15:                                             ; preds = %bb9
  tail call void (i8*, ...) @g(i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str40, i32 0, i32 0))
  br label %bb20

bb16:                                             ; preds = %bb9
  %i17 = call i32 @f(i32 -1037694186) #1
  %i18 = inttoptr i32 %i17 to i32 (i8**)*
  %i19 = tail call i32 %i18(i8** @gGlobalLock)
  br label %bb20

bb20:                                             ; preds = %bb16, %bb15
  ret void
}

declare i32 @f(i32)

declare void @g(i8*, ...)
