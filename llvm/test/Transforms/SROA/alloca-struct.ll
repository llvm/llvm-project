; RUN: opt < %s -sroa -S | FileCheck %s
; RUN: opt < %s -passes=sroa -S | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.RetValIntChar = type { i32, i8 }
%struct.RetValTwoInts = type { i32, i32 }
%struct.RetValOneIntTwoChar = type { i32, i8 }

; Tests that a struct of type {i32, i8} is scalarized by SROA.
; FIXME: SROA should skip scalarization since there is no scalar access.
; Currently scalarization happens due to the mismatch of allocated size
; and the actual structure size.
define i64 @test_struct_of_int_char(i1 zeroext %test, i64 ()* %p) {
; CHECK-LABEL: @test_struct_of_int_char(
; CHECK-NEXT:  entry:
; COM: Check that registers are used and alloca instructions are eliminated.
; CHECK-NOT:     alloca
; CHECK:       if.then:
; CHECK-NEXT:    br label [[RETURN:%.*]]
; CHECK:       if.end:
; CHECK-NEXT:    call i64 [[P:%.*]]()
; CHECK:    br label [[RETURN]]
; CHECK:       return:
; COM: Check there are more than one PHI nodes to select scalarized values.
; CHECK-COUNT-3: phi
; CHECK:         ret i64
;
entry:
  %retval = alloca %struct.RetValIntChar, align 4
  br i1 %test, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %x = getelementptr inbounds %struct.RetValIntChar, %struct.RetValIntChar* %retval, i32 0, i32 0
  store i32 0, i32* %x, align 4
  %y = getelementptr inbounds %struct.RetValIntChar, %struct.RetValIntChar* %retval, i32 0, i32 1
  store i8 0, i8* %y, align 4
  br label %return

if.end:                                           ; preds = %entry
  %call = call i64 %p()
  %0 = bitcast %struct.RetValIntChar* %retval to i64*
  store i64 %call, i64* %0, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %1 = bitcast %struct.RetValIntChar* %retval to i64*
  %2 = load i64, i64* %1, align 4
  ret i64 %2
}

; Test that the alloca of struct{int, int} will be scalarized by SROA.
define i64 @test_struct_of_two_int(i1 zeroext %test, i64 ()* %p) {
; CHECK-LABEL: @test_struct_of_two_int(
; CHECK-NEXT:  entry:
; CHECK-NOT:     alloca 
; CHECK:       if.then:
; CHECK-NEXT:    br label [[RETURN:%.*]]
; CHECK:       if.end:
; CHECK-NEXT:    call i64
; CHECK:       return:
; COM: Check that there are more than one PHI nodes to select the scalarized values.
; CHECK-COUNT-2: phi
; CHECK:         ret i64
;
entry:
  %retval = alloca %struct.RetValTwoInts, align 4
  br i1 %test, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %x = getelementptr inbounds %struct.RetValTwoInts, %struct.RetValTwoInts* %retval, i32 0, i32 0
  store i32 0, i32* %x, align 4
  %y = getelementptr inbounds %struct.RetValTwoInts, %struct.RetValTwoInts* %retval, i32 0, i32 1
  store i32 0, i32* %y, align 4
  br label %return

if.end:                                           ; preds = %entry
  %call = call i64 %p()
  %0 = bitcast %struct.RetValTwoInts* %retval to i64*
  store i64 %call, i64* %0, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %1 = bitcast %struct.RetValTwoInts* %retval to i64*
  %2 = load i64, i64* %1, align 4
  ret i64 %2
}

; Tests that allocated struct type is scalarized when non-constant values are
; stored into its fields.
define i64 @test_one_field_has_runtime_value(i1 zeroext %test, i64 ()* %p) {
; CHECK-LABEL: @test_one_field_has_runtime_value(
; CHECK-NEXT:  entry:
; CHECK-NOT:     alloca
; CHECK:         call void @srand
; CHECK:       if.then:
; CHECK-NEXT:    call i32 @rand()
; CHECK-NEXT:    br label
; CHECK:       if.end:
; CHECK-NEXT:    call i64
; CHECK:       return:
; CHECK-COUNT-2: phi i32
; CHECK:         ret i64
;
entry:
  %retval = alloca %struct.RetValTwoInts, align 4
  %call = call i64 @time(i64* null)
  %conv = trunc i64 %call to i32
  call void @srand(i32 %conv)
  br i1 %test, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %x = getelementptr inbounds %struct.RetValTwoInts, %struct.RetValTwoInts* %retval, i32 0, i32 0
  %call1 = call i32 @rand()
  store i32 %call1, i32* %x, align 4
  %y = getelementptr inbounds %struct.RetValTwoInts, %struct.RetValTwoInts* %retval, i32 0, i32 1
  store i32 1, i32* %y, align 4
  br label %return

if.end:                                           ; preds = %entry
  %call2 = call i64 %p()
  %0 = bitcast %struct.RetValTwoInts* %retval to i64*
  store i64 %call2, i64* %0, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %1 = bitcast %struct.RetValTwoInts* %retval to i64*
  %2 = load i64, i64* %1, align 4
  ret i64 %2
}

; Function Attrs: nounwind
declare void @srand(i32)

; Function Attrs: nounwind
declare i64 @time(i64*)

; Function Attrs: nounwind
declare i32 @rand()
