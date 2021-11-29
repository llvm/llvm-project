; RUN: opt -passes=globalopt -S < %s | FileCheck %s

@G1 = internal global i32 5
@G2 = internal global i32 5
@G3 = internal global i32 5
@G4 = internal global i32 5
@G5 = internal global i32 5

; CHECK-LABEL: @test1
define internal i32 @test1() norecurse {
; CHECK-NOT: @G1
  store i32 4, i32* @G1
  %a = load i32, i32* @G1
; CHECK: ret
  ret i32 %a
}

; The load comes before the store which makes @G2 live before the call.
; CHECK-LABEL: @test2
define internal i32 @test2() norecurse {
; CHECK-NOT: %G2
  %a = load i32, i32* @G2
  store i32 4, i32* @G2
; CHECK: ret
  ret i32 %a
}

; This global is indexed by a GEP - this makes it partial alias and we bail out.
; FIXME: We don't actually have to bail out in this case.

; CHECK-LABEL: @test3
define internal i32 @test3() norecurse {
; CHECK-NOT: %G3
  %x = getelementptr i32,i32* @G3, i32 0
  %a = load i32, i32* %x
  store i32 4, i32* @G3
; CHECK: ret
  ret i32 %a
}

; The global is casted away to a larger type then loaded. The store only partially
; covers the load, so we must not demote.

; CHECK-LABEL: @test4
define internal i32 @test4() norecurse {
; CHECK-NOT: %G4
  store i32 4, i32* @G4
  %x = bitcast i32* @G4 to i64*
  %a = load i64, i64* %x
  %b = trunc i64 %a to i32
; CHECK: ret
  ret i32 %b
}

; The global is casted away to a smaller type then loaded. This one is fine.

; CHECK-LABEL: @test5
define internal i32 @test5() norecurse {
; CHECK-NOT: @G5
  store i32 4, i32* @G5
  %x = bitcast i32* @G5 to i16*
  %a = load i16, i16* %x
  %b = zext i16 %a to i32
; CHECK: ret
  ret i32 %b
}

define i32 @main() norecurse {
  %a = call i32 @test1()
  %b = call i32 @test2()
  %c = call i32 @test3()
  %d = call i32 @test4()
  %e = call i32 @test5()

  %x = or i32 %a, %b
  %y = or i32 %x, %c
  %z = or i32 %y, %d
  %w = or i32 %z, %e
  ret i32 %w
}
