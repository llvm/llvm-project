; RUN: llc -mtriple=i386-apple-macosx < %s | FileCheck %s
; rdar://12396696

@JT = global [4 x i32] [i32 sub (i32 ptrtoint (ptr blockaddress(@h, %bb16) to i32), i32 ptrtoint (ptr blockaddress(@h, %bb9) to i32)), i32 sub (i32 ptrtoint (ptr blockaddress(@h, %bb15) to i32), i32 ptrtoint (ptr blockaddress(@h, %bb9) to i32)), i32 sub (i32 ptrtoint (ptr blockaddress(@h, %bb20) to i32), i32 ptrtoint (ptr blockaddress(@h, %bb16) to i32)), i32 sub (i32 ptrtoint (ptr blockaddress(@h, %bb20) to i32), i32 ptrtoint (ptr blockaddress(@h, %bb15) to i32))]
@gGlobalLock = external global ptr
@.str40 = external global [35 x i8]

; CHECK: _JT:
; CHECK-NOT: .long Ltmp{{[0-9]+}}-1
; CHECK-NOT: .long 1-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}
; CHECK: .long Ltmp{{[0-9]+}}-Ltmp{{[0-9]+}}

define void @h(ptr %arg) nounwind ssp {
bb:
  %i = alloca ptr, align 8
  store ptr %arg, ptr %i, align 8
  %i1 = load ptr, ptr %i, align 8
  %i3 = getelementptr { i32, i32 }, ptr %i1, i32 0, i32 0
  %i4 = load i32, ptr %i3, align 4
  %i5 = srem i32 %i4, 2
  %i6 = icmp slt i32 %i4, 2
  %i7 = select i1 %i6, i32 %i4, i32 %i5
  %i8 = icmp eq i32 %i7, 0
  br label %bb9

bb9:                                              ; preds = %bb
  %i10 = zext i1 %i8 to i32
  %i11 = getelementptr [4 x i32], ptr @JT, i32 0, i32 %i10
  %i12 = load i32, ptr %i11, align 4
  %i13 = add i32 %i12, ptrtoint (ptr blockaddress(@h, %bb9) to i32)
  %i14 = inttoptr i32 %i13 to ptr
  indirectbr ptr %i14, [label %bb15, label %bb16]

bb15:                                             ; preds = %bb9
  tail call void (ptr, ...) @g(ptr @.str40)
  br label %bb20

bb16:                                             ; preds = %bb9
  %i17 = call i32 @f(i32 -1037694186) #1
  %i18 = inttoptr i32 %i17 to ptr
  %i19 = tail call i32 %i18(ptr @gGlobalLock)
  br label %bb20

bb20:                                             ; preds = %bb16, %bb15
  ret void
}

declare i32 @f(i32)

declare void @g(ptr, ...)
