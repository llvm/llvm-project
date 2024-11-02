; RUN: opt -S -passes=instcombine < %s | FileCheck %s

; The last test needs this weird datalayout.
target datalayout = "i32:8:8"
; Without it, InstCombine will align the pointed on 4 Bytes
; The KnownBitsZero that result from the alignment allows to
; turn:
;    and i32 %mul, 255
; to:
;    and i32 %mul, 252
; The mask is no longer in the form 2^n-1  and this prevents the transformation.


; return mul(zext x, zext y) > MAX
define i32 @pr4917_1(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: @pr4917_1(
entry:
  %l = zext i32 %x to i64
  %r = zext i32 %y to i64
; CHECK-NOT: zext i32
  %mul64 = mul i64 %l, %r
; CHECK: [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 %y)
  %overflow = icmp ugt i64 %mul64, 4294967295
; CHECK: extractvalue { i32, i1 } [[MUL]], 1
  %retval = zext i1 %overflow to i32
  ret i32 %retval
}

; return mul(zext x, zext y) >= MAX+1
define i32 @pr4917_1a(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: @pr4917_1a(
entry:
  %l = zext i32 %x to i64
  %r = zext i32 %y to i64
; CHECK-NOT: zext i32
  %mul64 = mul i64 %l, %r
; CHECK: [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 %y)
  %overflow = icmp uge i64 %mul64, 4294967296
; CHECK: extractvalue { i32, i1 } [[MUL]], 1
  %retval = zext i1 %overflow to i32
  ret i32 %retval
}

; mul(zext x, zext y) > MAX
; mul(x, y) is used
define i32 @pr4917_2(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: @pr4917_2(
entry:
  %l = zext i32 %x to i64
  %r = zext i32 %y to i64
; CHECK-NOT: zext i32
  %mul64 = mul i64 %l, %r
; CHECK: [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 %y)
  %overflow = icmp ugt i64 %mul64, 4294967295
; CHECK-DAG: [[VAL:%.*]] = extractvalue { i32, i1 } [[MUL]], 0
  %mul32 = trunc i64 %mul64 to i32
; CHECK-DAG: [[OVFL:%.*]] = extractvalue { i32, i1 } [[MUL]], 1
  %retval = select i1 %overflow, i32 %mul32, i32 111
; CHECK: select i1 [[OVFL]], i32 [[VAL]]
  ret i32 %retval
}

; return mul(zext x, zext y) > MAX
; mul is used in non-truncate
define i64 @pr4917_3(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: @pr4917_3(
entry:
  %l = zext i32 %x to i64
  %r = zext i32 %y to i64
  %mul64 = mul i64 %l, %r
; CHECK-NOT: umul.with.overflow.i32
  %overflow = icmp ugt i64 %mul64, 4294967295
  %retval = select i1 %overflow, i64 %mul64, i64 111
  ret i64 %retval
}

; return mul(zext x, zext y) <= MAX
define i32 @pr4917_4(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: @pr4917_4(
entry:
  %l = zext i32 %x to i64
  %r = zext i32 %y to i64
; CHECK-NOT: zext i32
  %mul64 = mul i64 %l, %r
; CHECK: [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 %y)
  %overflow = icmp ule i64 %mul64, 4294967295
; CHECK: extractvalue { i32, i1 } [[MUL]], 1
; CHECK: xor
  %retval = zext i1 %overflow to i32
  ret i32 %retval
}

; return mul(zext x, zext y) < MAX+1
define i32 @pr4917_4a(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: @pr4917_4a(
entry:
  %l = zext i32 %x to i64
  %r = zext i32 %y to i64
; CHECK-NOT: zext i32
  %mul64 = mul i64 %l, %r
; CHECK: [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 %y)
  %overflow = icmp ult i64 %mul64, 4294967296
; CHECK: extractvalue { i32, i1 } [[MUL]], 1
; CHECK: xor
  %retval = zext i1 %overflow to i32
  ret i32 %retval
}

; operands of mul are of different size
define i32 @pr4917_5(i32 %x, i8 %y) nounwind {
; CHECK-LABEL: @pr4917_5(
entry:
  %l = zext i32 %x to i64
  %r = zext i8 %y to i64
; CHECK: [[Y:%.*]] = zext i8 %y to i32
  %mul64 = mul i64 %l, %r
  %overflow = icmp ugt i64 %mul64, 4294967295
  %mul32 = trunc i64 %mul64 to i32
; CHECK: [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 [[Y]])
; CHECK-DAG: [[VAL:%.*]] = extractvalue { i32, i1 } [[MUL]], 0
; CHECK-DAG: [[OVFL:%.*]] = extractvalue { i32, i1 } [[MUL]], 1
  %retval = select i1 %overflow, i32 %mul32, i32 111
; CHECK: select i1 [[OVFL]], i32 [[VAL]]
  ret i32 %retval
}

; mul(zext x, zext y) != zext trunc mul
define i32 @pr4918_1(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: @pr4918_1(
entry:
  %l = zext i32 %x to i64
  %r = zext i32 %y to i64
  %mul64 = mul i64 %l, %r
; CHECK: [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 %y)
  %part32 = trunc i64 %mul64 to i32
  %part64 = zext i32 %part32 to i64
  %overflow = icmp ne i64 %mul64, %part64
; CHECK: [[OVFL:%.*]] = extractvalue { i32, i1 } [[MUL:%.*]], 1
  %retval = zext i1 %overflow to i32
  ret i32 %retval
}

; mul(zext x, zext y) == zext trunc mul
define i32 @pr4918_2(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: @pr4918_2(
entry:
  %l = zext i32 %x to i64
  %r = zext i32 %y to i64
  %mul64 = mul i64 %l, %r
; CHECK: [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 %y)
  %part32 = trunc i64 %mul64 to i32
  %part64 = zext i32 %part32 to i64
  %overflow = icmp eq i64 %mul64, %part64
; CHECK: extractvalue { i32, i1 } [[MUL]]
  %retval = zext i1 %overflow to i32
; CHECK: xor
  ret i32 %retval
}

; zext trunc mul != mul(zext x, zext y)
define i32 @pr4918_3(i32 %x, i32 %y) nounwind {
; CHECK-LABEL: @pr4918_3(
entry:
  %l = zext i32 %x to i64
  %r = zext i32 %y to i64
  %mul64 = mul i64 %l, %r
; CHECK: [[MUL:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %x, i32 %y)
  %part32 = trunc i64 %mul64 to i32
  %part64 = zext i32 %part32 to i64
  %overflow = icmp ne i64 %part64, %mul64
; CHECK: extractvalue { i32, i1 } [[MUL]], 1
  %retval = zext i1 %overflow to i32
  ret i32 %retval
}

define <4 x i32> @pr20113(<4 x i16> %a, <4 x i16> %b) {
; CHECK-LABEL: @pr20113
; CHECK-NOT: mul.with.overflow
; CHECK: ret
  %vmovl.i.i726 = zext <4 x i16> %a to <4 x i32>
  %vmovl.i.i712 = zext <4 x i16> %b to <4 x i32>
  %mul.i703 = mul <4 x i32> %vmovl.i.i712, %vmovl.i.i726
  %tmp = icmp sge <4 x i32> %mul.i703, zeroinitializer
  %vcgez.i = sext <4 x i1> %tmp to <4 x i32>
  ret <4 x i32> %vcgez.i
}


@pr21445_data = external global i32
define i1 @pr21445(i8 %a) {
; CHECK-LABEL: @pr21445(
; CHECK-NEXT:  %[[umul:.*]] = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 %a, i8 ptrtoint (ptr @pr21445_data to i8))
; CHECK-NEXT:  %[[cmp:.*]] = extractvalue { i8, i1 } %[[umul]], 1
; CHECK-NEXT:  ret i1 %[[cmp]]
  %ext = zext i8 %a to i32
  %mul = mul i32 %ext, zext (i8 ptrtoint (ptr @pr21445_data to i8) to i32)
  %and = and i32 %mul, 255
  %cmp = icmp ne i32 %mul, %and
  ret i1 %cmp
}
