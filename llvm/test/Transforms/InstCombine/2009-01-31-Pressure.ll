; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {%B = add i8 %b, %x}
; PR2698

declare void @use1(i1)
declare void @use8(i8)

define void @test1(i8 %a, i8 %b, i8 %x) {
  %A = add i8 %a, %x
  %B = add i8 %b, %x
  %C = icmp eq i8 %A, %B
  call void @use1(i1 %C)
  ret void
}

define void @test2(i8 %a, i8 %b, i8 %x) {
  %A = add i8 %a, %x
  %B = add i8 %b, %x
  %C = icmp eq i8 %A, %B
  call void @use1(i1 %C)
  call void @use8(i8 %A)
  ret void
}
