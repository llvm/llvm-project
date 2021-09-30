; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define void @test4(i32 %n, ...) {
; CHECK: swm $a1, 4($sp), 7
  call void asm sideeffect "", ""()
  ret void
}

%struct.bar = type { i32, i32, i32 }

define void @square(%struct.bar* %ints) {
; CHECK: lwm $a1, 0($a0), 2
  %a = getelementptr inbounds %struct.bar, %struct.bar* %ints, i32 0, i32 0
  %1 = load i32, i32* %a, align 4
  %b = getelementptr inbounds %struct.bar, %struct.bar* %ints, i32 0, i32 1
  %2 = load i32, i32* %b, align 4
  %add = add nsw i32 %2, %1
  %c = getelementptr inbounds %struct.bar, %struct.bar* %ints, i32 0, i32 2
  store i32 %add, i32* %c, align 4
  ret void
}
