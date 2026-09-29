; RUN: opt -passes=constmerge -S < %s | FileCheck %s
; Test which corresponding x and y are merged and that unnamed_addr
; is correctly set.

declare void @zed(ptr, ptr)

%struct.foobar = type { i32 }

@test1.x = internal constant %struct.foobar { i32 1 }
@test1.y = constant %struct.foobar { i32 1 }

@test2.x = internal constant %struct.foobar { i32 2 }
@test2.y = unnamed_addr constant %struct.foobar { i32 2 }

@test3.x = internal unnamed_addr constant %struct.foobar { i32 3 }
@test3.y = constant %struct.foobar { i32 3 }

@test4.x = internal unnamed_addr constant %struct.foobar { i32 4 }
@test4.y = unnamed_addr constant %struct.foobar { i32 4 }


; CHECK:      %struct.foobar = type { i32 }
; CHECK-NOT: @
; CHECK: @test1.x = internal constant %struct.foobar { i32 1 }
; CHECK-NEXT: @test1.y = constant %struct.foobar { i32 1 }
; CHECK-NEXT: @test2.y = constant %struct.foobar { i32 2 }
; CHECK-NEXT: @test3.y = constant %struct.foobar { i32 3 }
; CHECK-NEXT: @test4.y = unnamed_addr constant %struct.foobar { i32 4 }
; CHECK-NOT: @
; CHECK: declare void @zed(ptr, ptr)

define i32 @main() {
entry:
  call void @zed(ptr @test1.x, ptr @test1.y)
  call void @zed(ptr @test2.x, ptr @test2.y)
  call void @zed(ptr @test3.x, ptr @test3.y)
  call void @zed(ptr @test4.x, ptr @test4.y)
  ret i32 0
}

