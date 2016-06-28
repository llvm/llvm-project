;RUN: llc < %s -mcpu=core-avx2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.struct_1 = type { i24 }

@a = global i8 0, align 1
@b = global i8 0, align 1
@d = global i8 0, align 1
@e = global i8 0, align 1
@c = global %struct.struct_1 zeroinitializer, align 4

; Function Attrs: norecurse nounwind uwtable
define void @_Z3fn1v() #0 {
  %bf.load = load i32, i32* bitcast (%struct.struct_1* @c to i32*), align 4
  %bf.shl = shl i32 %bf.load, 8
  %bf.ashr = ashr exact i32 %bf.shl, 8
  %tobool4 = icmp ne i32 %bf.ashr, 0
  %conv = zext i1 %tobool4 to i32
  %x1 = load i8, i8* @e, align 1
  %conv6 = zext i8 %x1 to i32
  %add = add nuw nsw i32 %conv, %conv6
  %tobool7 = icmp ne i32 %add, 0
  %frombool = zext i1 %tobool7 to i8
  store i8 %frombool, i8* @b, align 1
  %tobool14 = icmp ne i32 %bf.shl, 0
  %frombool15 = zext i1 %tobool14 to i8
  store i8 %frombool15, i8* @d, align 1
  ret void
}