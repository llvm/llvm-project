; RUN: llc < %s -march=xcore | FileCheck %s

declare void @llvm.xcore.setv.p1i8(i8 addrspace(1)* %r, i8* %p)
declare i8* @llvm.xcore.waitevent()
declare i8* @llvm.xcore.checkevent(i8*)
declare void @llvm.xcore.clre()

define i32 @f(i8 addrspace(1)* %r) nounwind {
; CHECK: f:
entry:
; CHECK: clre
  call void @llvm.xcore.clre()
  call void @llvm.xcore.setv.p1i8(i8 addrspace(1)* %r, i8* blockaddress(@f, %L1))
  call void @llvm.xcore.setv.p1i8(i8 addrspace(1)* %r, i8* blockaddress(@f, %L2))
  %goto_addr = call i8* @llvm.xcore.waitevent()
; CHECK: waiteu
  indirectbr i8* %goto_addr, [label %L1, label %L2]
L1:
  br label %ret
L2:
  br label %ret
ret:
  %retval = phi i32 [1, %L1], [2, %L2]
  ret i32 %retval
}

define i32 @g(i8 addrspace(1)* %r) nounwind {
; CHECK: g:
entry:
; CHECK: clre
  call void @llvm.xcore.clre()
  call void @llvm.xcore.setv.p1i8(i8 addrspace(1)* %r, i8* blockaddress(@f, %L1))
  %goto_addr = call i8* @llvm.xcore.checkevent(i8 *blockaddress(@f, %L2))
; CHECK: setsr 1
; CHECK: clrsr 1
  indirectbr i8* %goto_addr, [label %L1, label %L2]
L1:
  br label %ret
L2:
  br label %ret
ret:
  %retval = phi i32 [1, %L1], [2, %L2]
  ret i32 %retval
}
