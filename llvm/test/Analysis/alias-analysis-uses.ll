; RUN: opt -debug-pass=Executions -globals-aa -functionattrs -disable-output < %s 2>&1 | FileCheck %s

; CHECK: Executing Pass 'Globals Alias Analysis'
; CHECK-NOT: Freeing Pass 'Globals Alias Analysis'
; CHECK: Executing Pass 'Deduce function attributes'
; CHECK: Freeing Pass 'Globals Alias Analysis'

define void @test(i8* %p) {
  ret void
}
