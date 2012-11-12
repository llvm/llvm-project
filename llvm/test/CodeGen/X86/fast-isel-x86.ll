; RUN: llc -fast-isel -O0 -mcpu=generic -mtriple=i386-apple-darwin10 -relocation-model=pic < %s | FileCheck %s

; This should use flds to set the return value.
; CHECK: test0:
; CHECK: flds
; CHECK: ret
@G = external global float
define float @test0() nounwind {
  %t = load float* @G
  ret float %t
}

; This should pop 4 bytes on return.
; CHECK: test1:
; CHECK: ret $4
define void @test1({i32, i32, i32, i32}* sret %p) nounwind {
  store {i32, i32, i32, i32} zeroinitializer, {i32, i32, i32, i32}* %p
  ret void
}

; Properly initialize the pic base.
; CHECK: test2:
; CHECK-NOT: HHH
; CHECK: call{{.*}}L2$pb
; CHECK-NEXT: L2$pb:
; CHECK-NEXT: pop
; CHECK: HHH
; CHECK: ret
@HHH = external global i32
define i32 @test2() nounwind {
  %t = load i32* @HHH
  ret i32 %t
}

; Check that we fast-isel sret, and handle the callee-pops behavior correctly.
%struct.a = type { i64, i64, i64 }
define void @test3() nounwind ssp {
entry:
  %tmp = alloca %struct.a, align 8
  call void @test3sret(%struct.a* sret %tmp)
  ret void
; CHECK: test3:
; CHECK: subl $44
; CHECK: leal 16(%esp)
; CHECK: calll _test3sret
; CHECK: addl $40
}
declare void @test3sret(%struct.a* sret)

; Check that fast-isel sret works with fastcc (and does not callee-pop)
define void @test4() nounwind ssp {
entry:
  %tmp = alloca %struct.a, align 8
  call fastcc void @test4fastccsret(%struct.a* sret %tmp)
  ret void
; CHECK: test4:
; CHECK: subl $28
; CHECK: leal (%esp), %ecx
; CHECK: calll _test4fastccsret
; CHECK: addl $28
}
declare fastcc void @test4fastccsret(%struct.a* sret)
