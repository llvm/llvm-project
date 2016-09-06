; RUN: llc -verify-machineinstrs < %s
; REQUIRES: default_triple

; This used to cause a crash.  A standard load is converted to a pre-increment
; load.  Later the pre-increment load is combined with a subsequent SRL to
; produce a smaller load.  This transform invalidly created a standard load
; and propagated the produced value into uses of both produced values of the
; pre-increment load.  The result was a crash when attempting to process an
; add with a token-chain operand.

%struct.Info = type { i32, i32, i8*, i8*, i8*, [32 x i8*], i64, [32 x i64], i64, i64, i64, [32 x i64] }
%struct.S1847 = type { [12 x i8], [4 x i8], [8 x i8], [4 x i8], [8 x i8], [2 x i8], i8, [4 x i64], i8, [3 x i8], [4 x i8], i8, i16, [4 x %struct.anon.76], i16, i8, i8* }
%struct.anon.76 = type { i32 }
@info = common global %struct.Info zeroinitializer, align 8
@fails = common global i32 0, align 4
@a1847 = external global [5 x %struct.S1847]
define void @test1847() nounwind {
entry:
  %j = alloca i32, align 4
  %0 = load i64, i64* getelementptr inbounds (%struct.Info, %struct.Info* @info, i32 0, i32 8), align 8
  %1 = load i32, i32* @fails, align 4
  %bf.load1 = load i96, i96* bitcast (%struct.S1847* getelementptr inbounds ([5 x %struct.S1847], [5 x %struct.S1847]* @a1847, i32 0, i64 2) to i96*), align 8
  %bf.clear2 = and i96 %bf.load1, 302231454903657293676543
  %bf.set3 = or i96 %bf.clear2, -38383394772764476296921088
  store i96 %bf.set3, i96* bitcast (%struct.S1847* getelementptr inbounds ([5 x %struct.S1847], [5 x %struct.S1847]* @a1847, i32 0, i64 2) to i96*), align 8
  %2 = load i32, i32* %j, align 4
  %3 = load i32, i32* %j, align 4
  %inc11 = add nsw i32 %3, 1
  store i32 %inc11, i32* %j, align 4
  %bf.load15 = load i96, i96* bitcast (%struct.S1847* getelementptr inbounds ([5 x %struct.S1847], [5 x %struct.S1847]* @a1847, i32 0, i64 2) to i96*), align 8
  %bf.clear16 = and i96 %bf.load15, -18446744069414584321
  %bf.set17 = or i96 %bf.clear16, 18446743532543672320
  store i96 %bf.set17, i96* bitcast (%struct.S1847* getelementptr inbounds ([5 x %struct.S1847], [5 x %struct.S1847]* @a1847, i32 0, i64 2) to i96*), align 8
  ret void
}
