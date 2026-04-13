; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK-LABEL:   llvm.func @test(
; CHECK-SAME:      %[[VAL_0:.*]]: i16 {llvm.noundef, llvm.signext}) -> (i16 {llvm.signext}) attributes {personality = @__gxx_personality_v0} {
define signext i16 @test(i16 noundef signext %0) personality ptr @__gxx_personality_v0 {
  ; CHECK:           %[[VAL_3:.*]] = llvm.invoke @somefunc(%[[VAL_0]]) to ^bb2 unwind ^bb1 : (i16 {llvm.noundef, llvm.signext}) -> (i16 {llvm.signext})
  %2 = invoke signext i16 @somefunc(i16 noundef signext %0)
          to label %7 unwind label %3

3:                                                ; preds = %1
  %4 = landingpad { ptr, i32 }
          catch ptr null
  %5 = extractvalue { ptr, i32 } %4, 0
  %6 = tail call ptr @__cxa_begin_catch(ptr %5) #2
  tail call void @__cxa_end_catch()
  br label %7

7:                                                ; preds = %1, %3
  %8 = phi i16 [ 0, %3 ], [ %2, %1 ]
  ret i16 %8
}

declare noundef signext i16 @somefunc(i16 noundef signext)
declare i32 @__gxx_personality_v0(...)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
