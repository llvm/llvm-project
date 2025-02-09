; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

target triple = "hexagon"

%s.0 = type { ptr, ptr, ptr }
%s.1 = type { ptr, ptr, ptr }
%s.2 = type { ptr, ptr, ptr, ptr, ptr, ptr }
%s.3 = type { ptr, ptr, ptr }
%s.4 = type { ptr, ptr, ptr }
%s.5 = type { ptr, ptr, i32 }

; Function Attrs: nounwind optsize
declare zeroext i1 @f0(ptr) #0 align 2

; Function Attrs: nounwind optsize
declare zeroext i1 @f1(ptr) #0 align 2

; Function Attrs: optsize
declare hidden void @f2(ptr noalias nocapture sret(i32), i32) #1 align 2

; Function Attrs: optsize
declare hidden void @f3(ptr noalias nocapture sret(i32), i32) #1 align 2

; Function Attrs: optsize
declare hidden void @f4(ptr noalias nocapture sret(i32), i32) #1 align 2

; Function Attrs: optsize
declare hidden void @f5(ptr noalias nocapture sret(i32), i32) #1 align 2

; Function Attrs: optsize
declare hidden void @f6(ptr noalias nocapture sret(i32), i32) #1 align 2

; Function Attrs: optsize
declare hidden void @f7(ptr noalias nocapture sret(i32), i32) #1 align 2

; Function Attrs: optsize
declare zeroext i1 @f8(ptr, ptr, i64) #1 align 2

; Function Attrs: nounwind optsize
declare ptr @f9(ptr nocapture readonly) #0 align 2

; Function Attrs: optsize
define void @f10(ptr %a0, ptr dereferenceable(64) %a1) #1 align 2 {
b0:
  %v0 = alloca %s.0, align 4
  %v1 = alloca %s.1, align 4
  %v2 = alloca %s.2, align 4
  %v3 = alloca %s.3, align 4
  %v4 = alloca %s.4, align 4
  %v5 = alloca %s.5, align 8
  br i1 undef, label %b34, label %b1

b1:                                               ; preds = %b0
  br i1 undef, label %b3, label %b2

b2:                                               ; preds = %b1
  %v6 = ptrtoint ptr %v0 to i32
  %v7 = zext i32 %v6 to i64
  %v8 = shl nuw i64 %v7, 32
  %f2.ext = zext i32 ptrtoint (ptr @f2 to i32) to i64
  %v9 = or i64 %v8, %f2.ext
  %v10 = ptrtoint ptr %v4 to i32
  %v11 = zext i32 %v10 to i64
  %v12 = shl nuw i64 %v11, 32
  %f5.ext = zext i32 ptrtoint (ptr @f5 to i32) to i64
  %v13 = or i64 %v12, %f5.ext
  %v14 = ptrtoint ptr %v5 to i32
  %v15 = zext i32 %v14 to i64
  %v16 = shl nuw i64 %v15, 32
  %f6.ext = zext i32 ptrtoint (ptr @f6 to i32) to i64
  %v17 = or i64 %v16, %f6.ext
  %v18 = ptrtoint ptr %v1 to i32
  %v19 = zext i32 %v18 to i64
  %v20 = shl nuw i64 %v19, 32
  %f3.ext = zext i32 ptrtoint (ptr @f3 to i32) to i64
  %v21 = or i64 %v20, %f3.ext
  %v22 = ptrtoint ptr %v2 to i32
  %v23 = zext i32 %v22 to i64
  %v24 = shl nuw i64 %v23, 32
  %f4.ext = zext i32 ptrtoint (ptr @f4 to i32) to i64
  %v25 = or i64 %v24, %f4.ext
  %v26 = ptrtoint ptr %v3 to i32
  %v27 = zext i32 %v26 to i64
  %v28 = shl nuw i64 %v27, 32
  %f7.ext = zext i32 ptrtoint (ptr @f7 to i32) to i64
  %v29 = or i64 %v28, %f7.ext
  %v30 = call ptr @f9(ptr nonnull null) #1
  br i1 undef, label %b5, label %b4

b3:                                               ; preds = %b1
  unreachable

b4:                                               ; preds = %b2
  store ptr null, ptr null, align 4
  %v31 = call zeroext i1 @f0(ptr null) #0
  br i1 %v31, label %b6, label %b32

b5:                                               ; preds = %b2
  unreachable

b6:                                               ; preds = %b4
  br i1 undef, label %b7, label %b32

b7:                                               ; preds = %b6
  br i1 undef, label %b8, label %b32

b8:                                               ; preds = %b7
  br i1 undef, label %b9, label %b32

b9:                                               ; preds = %b8
  br i1 undef, label %b10, label %b32

b10:                                              ; preds = %b9
  %v32 = call zeroext i1 @f1(ptr null) #0
  br i1 %v32, label %b11, label %b32

b11:                                              ; preds = %b10
  br i1 undef, label %b13, label %b12

b12:                                              ; preds = %b11
  unreachable

b13:                                              ; preds = %b11
  %v33 = call zeroext i1 @f0(ptr undef) #0
  br i1 %v33, label %b14, label %b32

b14:                                              ; preds = %b13
  br i1 undef, label %b16, label %b15

b15:                                              ; preds = %b14
  unreachable

b16:                                              ; preds = %b14
  %v34 = call zeroext i1 @f1(ptr null) #0
  br i1 %v34, label %b18, label %b17

b17:                                              ; preds = %b16
  unreachable

b18:                                              ; preds = %b16
  br i1 undef, label %b19, label %b32

b19:                                              ; preds = %b18
  br i1 undef, label %b26, label %b20

b20:                                              ; preds = %b19
  br i1 undef, label %b22, label %b21

b21:                                              ; preds = %b20
  br i1 undef, label %b23, label %b32

b22:                                              ; preds = %b20
  unreachable

b23:                                              ; preds = %b21
  br i1 undef, label %b24, label %b32

b24:                                              ; preds = %b23
  %v35 = call zeroext i1 @f8(ptr nonnull %a1, ptr undef, i64 undef) #1
  br i1 %v35, label %b25, label %b32

b25:                                              ; preds = %b24
  %v36 = call zeroext i1 @f8(ptr nonnull %a1, ptr undef, i64 %v9) #1
  unreachable

b26:                                              ; preds = %b19
  br i1 undef, label %b27, label %b32

b27:                                              ; preds = %b26
  br i1 undef, label %b28, label %b32

b28:                                              ; preds = %b27
  br i1 undef, label %b31, label %b29

b29:                                              ; preds = %b28
  %v37 = call zeroext i1 @f8(ptr nonnull %a1, ptr null, i64 %v21) #1
  %v38 = call zeroext i1 @f8(ptr nonnull %a1, ptr undef, i64 %v25) #1
  br i1 %v38, label %b30, label %b32

b30:                                              ; preds = %b29
  %v39 = call zeroext i1 @f8(ptr nonnull %a1, ptr undef, i64 %v29) #1
  unreachable

b31:                                              ; preds = %b28
  %v40 = call zeroext i1 @f8(ptr nonnull %a1, ptr null, i64 %v13) #1
  %v41 = call zeroext i1 @f8(ptr nonnull %a1, ptr undef, i64 %v17) #1
  br i1 %v41, label %b33, label %b32

b32:                                              ; preds = %b31, %b29, %b27, %b26, %b24, %b23, %b21, %b18, %b13, %b10, %b9, %b8, %b7, %b6, %b4
  unreachable

b33:                                              ; preds = %b31
  store ptr %a0, ptr undef, align 4
  unreachable

b34:                                              ; preds = %b0
  ret void
}

attributes #0 = { nounwind optsize }
attributes #1 = { optsize }
