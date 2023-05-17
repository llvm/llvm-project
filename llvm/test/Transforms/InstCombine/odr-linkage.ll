; RUN: opt < %s -passes=instcombine -S | grep "ret i32 10"

@g1 = available_externally constant i32 1
@g2 = linkonce_odr constant i32 2
@g3 = weak_odr constant i32 3
@g4 = internal constant i32 4

define i32 @test() {
  %A = load i32, ptr @g1
  %B = load i32, ptr @g2
  %C = load i32, ptr @g3
  %D = load i32, ptr @g4
  
  %a = add i32 %A, %B
  %b = add i32 %a, %C
  %c = add i32 %b, %D
  ret i32 %c
}
   
