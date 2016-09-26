; RUN: llc -fast-isel -O0 -mcpu=generic -mtriple=i386-apple-darwin10 -relocation-model=pic < %s | FileCheck %s
; RUN: llc -fast-isel -O0 -mcpu=generic -mtriple=i386-apple-darwin10 -relocation-model=pic < %s -fast-isel-verbose 2>&1 >/dev/null | FileCheck -check-prefix=STDERR -allow-empty %s

; This should use flds to set the return value.
; CHECK-LABEL: test0:
; CHECK: flds
; CHECK: retl
@G = external global float
define float @test0() nounwind {
  %t = load float, float* @G
  ret float %t
}

; This should pop 4 bytes on return.
; CHECK-LABEL: test1:
; CHECK: retl $4
define void @test1({i32, i32, i32, i32}* sret %p) nounwind {
  store {i32, i32, i32, i32} zeroinitializer, {i32, i32, i32, i32}* %p
  ret void
}

; This should pop 8 bytes on return.
; CHECK-LABEL: thiscallfun:
; CHECK: retl $8
define x86_thiscallcc i32 @thiscallfun(i32* %this, i32 %a, i32 %b) nounwind {
; STDERR-NOT: FastISel missed terminator: ret i32 12345
  ret i32 12345
}

; Here, the callee pop doesn't fit the 16 bit immediate -- see x86-big-ret.ll
; This checks that -fast-isel doesn't miscompile this.
; CHECK-LABEL: thiscall_large:
; CHECK:      popl %ecx
; CHECK-NEXT: addl $65536, %esp
; CHECK-NEXT: pushl %ecx
; CHECK-NEXT: retl
define x86_thiscallcc void @thiscall_large(i32* %this, [65533 x i8]* byval %b) nounwind {
  ret void
}

; This should pop 4 bytes on return.
; CHECK-LABEL: stdcallfun:
; CHECK: retl $4
define x86_stdcallcc i32 @stdcallfun(i32 %a) nounwind {
; STDERR-NOT: FastISel missed terminator: ret i32 54321
  ret i32 54321
}

; Properly initialize the pic base.
; CHECK-LABEL: test2:
; CHECK-NOT: HHH
; CHECK: call{{.*}}L5$pb
; CHECK-NEXT: L5$pb:
; CHECK-NEXT: pop
; CHECK: HHH
; CHECK: retl
@HHH = external global i32
define i32 @test2() nounwind {
  %t = load i32, i32* @HHH
  ret i32 %t
}

; Check that we fast-isel sret, and handle the callee-pops behavior correctly.
%struct.a = type { i64, i64, i64 }
define void @test3() nounwind ssp {
entry:
  %tmp = alloca %struct.a, align 8
  call void @test3sret(%struct.a* sret %tmp)
  ret void
; CHECK-LABEL: test3:
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
; CHECK-LABEL: test4:
; CHECK: subl $28
; CHECK: movl %esp, %ecx
; CHECK: calll _test4fastccsret
; CHECK: addl $28
}
declare fastcc void @test4fastccsret(%struct.a* sret)


; Check that fast-isel cleans up when it fails to lower a call instruction.
define void @test5() {
entry:
  %call = call i32 @test5dllimport(i32 42)
  ret void
; CHECK-LABEL: test5:
; Local value area is still there:
; CHECK: movl $42, {{%[a-z]+}}
; Fast-ISel's arg push is not here:
; CHECK-NOT: movl $42, (%esp)
; SDag-ISel's arg push:
; CHECK: movl %esp, [[REGISTER:%[a-z]+]]
; CHECK: movl $42, ([[REGISTER]])
; CHECK: movl L_test5dllimport$non_lazy_ptr-L8$pb(%eax), %eax

}
declare dllimport i32 @test5dllimport(i32)
