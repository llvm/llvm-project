; RUN: opt -passes=globalopt < %s -S | FileCheck %s

; Check that we don't try to set a global initializer to a value of a different type.
; In this case, we were trying to set @0's initializer to be i32* null.

%0 = type { i32* }

@0 = internal global %0* null
; CHECK: global %0 undef

define void @a() {
  %1 = tail call i8* @_Znwm(i64 8)
  %2 = bitcast i8* %1 to %0*
  %3 = getelementptr inbounds %0, %0* %2, i64 0, i32 0
  store i32* null, i32** %3, align 8
  store i8* %1, i8** bitcast (%0** @0 to i8**), align 8
  %4 = load i64*, i64** bitcast (%0** @0 to i64**), align 8
  %5 = load atomic i64, i64* %4 acquire, align 8
  ret void
}

declare i8* @_Znwm(i64)
