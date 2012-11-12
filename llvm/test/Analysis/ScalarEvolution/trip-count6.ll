; RUN: opt < %s -analyze -scalar-evolution \
; RUN:  | grep "max backedge-taken count is 1$"

@mode_table = global [4 x i32] zeroinitializer          ; <[4 x i32]*> [#uses=1]

define i8 @f() {
entry:
  tail call i32 @fegetround( )          ; <i32>:0 [#uses=1]
  br label %bb

bb:             ; preds = %bb4, %entry
  %mode.0 = phi i8 [ 0, %entry ], [ %indvar.next, %bb4 ]                ; <i8> [#uses=4]
  zext i8 %mode.0 to i32                ; <i32>:1 [#uses=1]
  getelementptr [4 x i32]* @mode_table, i32 0, i32 %1           ; <i32*>:2 [#uses=1]
  load i32* %2, align 4         ; <i32>:3 [#uses=1]
  icmp eq i32 %3, %0            ; <i1>:4 [#uses=1]
  br i1 %4, label %bb1, label %bb2

bb1:            ; preds = %bb
  ret i8 %mode.0

bb2:            ; preds = %bb
  icmp eq i8 %mode.0, 1         ; <i1>:5 [#uses=1]
  br i1 %5, label %bb5, label %bb4

bb4:            ; preds = %bb2
  %indvar.next = add i8 %mode.0, 1              ; <i8> [#uses=1]
  br label %bb

bb5:            ; preds = %bb2
  tail call void @raise_exception( ) noreturn 
  unreachable
}

declare i32 @fegetround()

declare void @raise_exception() noreturn 
