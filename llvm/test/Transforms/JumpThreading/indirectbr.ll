; RUN: opt -S < %s -passes=jump-threading | FileCheck %s

; Keep block addresses alive.
@addresses = constant [4 x ptr] [
  ptr blockaddress(@test1, %L1), ptr blockaddress(@test1, %L2),
  ptr blockaddress(@test2, %L1), ptr blockaddress(@test2, %L2)
]

declare void @bar()
declare void @baz()



; Check basic jump threading for indirectbr instructions.

; CHECK: void @test1
; CHECK: br i1 %tobool, label %L1, label %indirectgoto
; CHECK-NOT: if.else:
; CHECK: L1:
; CHECK: indirectbr ptr %address, [label %L1, label %L2]
define void @test1(i32 %i, ptr %address) nounwind {
entry:
  %rem = srem i32 %i, 2
  %tobool = icmp ne i32 %rem, 0
  br i1 %tobool, label %indirectgoto, label %if.else

if.else:                                          ; preds = %entry
  br label %indirectgoto

L1:                                               ; preds = %indirectgoto
  call void @bar()
  ret void

L2:                                               ; preds = %indirectgoto
  call void @baz()
  ret void

indirectgoto:                                     ; preds = %if.else, %entry
  %indirect.goto.dest = phi ptr [ %address, %if.else ], [ blockaddress(@test1, %L1), %entry ]
  indirectbr ptr %indirect.goto.dest, [label %L1, label %L2]
}


; Check constant folding of indirectbr

; CHECK: void @test2
; CHECK: entry:
; CHECK-NEXT: br label %L1
; CHECK: L1:
; CHECK-NEXT: call void @bar
; CHECK-NEXT: ret void
define void @test2() nounwind {
entry:
  indirectbr ptr blockaddress(@test2, %L1), [label %L1, label %L2]

L1:                                               ; preds = %indirectgoto
  call void @bar()
  ret void

L2:                                               ; preds = %indirectgoto
  call void @baz()
  ret void
}


; PR4151
; Don't merge address-taken blocks.
@.str = private unnamed_addr constant [4 x i8] c"%p\0A\00"

; CHECK-LABEL: @test3(
; CHECK: __here:
; CHECK: blockaddress(@test3, %__here)
; CHECK: __here1:
; CHECK: blockaddress(@test3, %__here1)
; CHECK: __here3:
; CHECK: blockaddress(@test3, %__here3)
define void @test3() nounwind ssp noredzone {
entry:
  br label %__here

__here:                                           ; preds = %entry
  %call = call i32 (...) @printf(ptr @.str, i64 ptrtoint (ptr blockaddress(@test3, %__here) to i64)) nounwind noredzone
  br label %__here1

__here1:                                          ; preds = %__here
  %call2 = call i32 (...) @printf(ptr @.str, i64 ptrtoint (ptr blockaddress(@test3, %__here1) to i64)) nounwind noredzone
  br label %__here3

__here3:                                          ; preds = %__here1
  %call4 = call i32 (...) @printf(ptr @.str, i64 ptrtoint (ptr blockaddress(@test3, %__here3) to i64)) nounwind noredzone
  ret void
}

declare i32 @printf(...) noredzone
