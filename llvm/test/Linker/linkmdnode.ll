; RUN: llvm-as < %s > %t.bc
; RUN: llvm-as < %p/linkmdnode2.ll > %t2.bc
; RUN: llvm-link %t.bc %t2.bc


!21 = metadata !{i32 42, metadata !"foobar"}

declare i8 @llvm.something(metadata %a)
define void @foo() {
  %x = call i8 @llvm.something(metadata !21)
  ret void
}

