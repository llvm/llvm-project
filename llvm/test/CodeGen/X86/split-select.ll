; RUN: llvm-as < %s | llc -march=x86-64 | grep test | count 1

define void @foo(i1 %c, <2 x i16> %a, <2 x i16> %b, <2 x i16>* %p) {
  %x = select i1 %c, <2 x i16> %a, <2 x i16> %b
  store <2 x i16> %x, <2 x i16>* %p
  ret void
}
