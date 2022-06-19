; RUN: opt < %s -passes=instcombine -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct._my_struct = type <{ [12 x i8], [4 x i8] }>

@initval = common global %struct._my_struct zeroinitializer, align 1

; InstCombine will try to change the %struct._my_struct alloca into an
; allocation of an i96 because of the bitcast to create %2. That's not valid,
; as the other 32 bits of the structure still feed into the return value
define { i64, i64 } @function(i32 %x, i32 %y, i32 %z) nounwind {
; CHECK-LABEL: @function(
; CHECK-NEXT: entry:
; CHECK-NEXT: %retval = alloca %struct._my_struct, align 8
; CHECK-NOT: bitcast i96* %retval to %struct._my_struct*
entry:
  %retval = alloca %struct._my_struct, align 8
  %k.sroa.0.0.copyload = load i96, i96* bitcast (%struct._my_struct* @initval to i96*), align 1
  %k.sroa.1.12.copyload = load i32, i32* bitcast ([4 x i8]* getelementptr inbounds (%struct._my_struct, %struct._my_struct* @initval, i64 0, i32 1) to i32*), align 1
  %0 = zext i32 %x to i96
  %bf.value = shl nuw nsw i96 %0, 6
  %bf.clear = and i96 %k.sroa.0.0.copyload, -288230376151711744
  %1 = zext i32 %y to i96
  %bf.value2 = shl nuw nsw i96 %1, 32
  %bf.shl3 = and i96 %bf.value2, 288230371856744448
  %bf.value.masked = and i96 %bf.value, 4294967232
  %2 = zext i32 %z to i96
  %bf.value8 = and i96 %2, 63
  %bf.clear4 = or i96 %bf.shl3, %bf.value.masked
  %bf.set5 = or i96 %bf.clear4, %bf.value8
  %bf.set10 = or i96 %bf.set5, %bf.clear
  %retval.0.cast7 = bitcast %struct._my_struct* %retval to i96*
  store i96 %bf.set10, i96* %retval.0.cast7, align 8
  %retval.12.idx8 = getelementptr inbounds %struct._my_struct, %struct._my_struct* %retval, i64 0, i32 1
  %retval.12.cast9 = bitcast [4 x i8]* %retval.12.idx8 to i32*
  store i32 %k.sroa.1.12.copyload, i32* %retval.12.cast9, align 4
  %trunc = trunc i96 %bf.set10 to i64
  %.fca.0.insert = insertvalue { i64, i64 } undef, i64 %trunc, 0
  %retval.8.idx12 = getelementptr inbounds %struct._my_struct, %struct._my_struct* %retval, i64 0, i32 0, i64 8
  %retval.8.cast13 = bitcast i8* %retval.8.idx12 to i64*
  %retval.8.load14 = load i64, i64* %retval.8.cast13, align 8
  %.fca.1.insert = insertvalue { i64, i64 } %.fca.0.insert, i64 %retval.8.load14, 1
  ret { i64, i64 } %.fca.1.insert
}
