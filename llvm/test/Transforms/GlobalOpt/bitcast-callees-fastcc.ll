; RUN:  opt < %s -globalopt -S | FileCheck %s

; CHECK-LABEL: define internal fastcc i32 @foo() unnamed_addr
define internal i32 @foo() {
   ret i32 8
}

; CHECK-LABEL: define void @test_bitcast_callees2() local_unnamed_addr {
define void @test_bitcast_callees2() {
  ; CHECK: %1 = call fastcc float bitcast (i32 ()* @foo to float ()*)()
  call float bitcast (i32()* @foo to float()*)()
  ; CHECK-NEXT: %2 = call fastcc float bitcast (i32 ()* @foo to float ()*)()
  call float bitcast (i32()* @foo to float()*)()
  ; CHECK-NEXT: %3 = call fastcc i8 bitcast (i32 ()* @foo to i8 ()*)()
  call i8 bitcast (i32()* @foo to i8()*)()
  ret void
}

