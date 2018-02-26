; RUN: llc -mtriple=x86_64-- < %s

define void @dw2102_i2c_transfer() nounwind {
entry:
  br label %bb

bb:                                               ; preds = %bb, %entry
  %z = phi i64 [ 0, %entry ], [ %z3, %bb ]
  %z1 = phi i16 [ undef, %entry ], [ %z6, %bb ]
  %z2 = phi i32 [ 0, %entry ], [ %z8, %bb ]
  %z3 = add i64 %z, 1
  %z4 = zext i16 %z1 to i32
  %z5 = add nsw i32 %z4, %z2
  %z6 = trunc i32 %z5 to i16
  call fastcc void @dw210x_op_rw(i16 zeroext %z6)
  %z7 = getelementptr i8, i8* null, i64 %z
  store i8 undef, i8* %z7, align 1
  %z8 = add nsw i32 %z2, 1
  br label %bb
}

declare fastcc void @dw210x_op_rw(i16 zeroext) nounwind
