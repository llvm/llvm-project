; RUN: llvm-as < %s | llc -march=ppc32 | not grep .comm.*X,0

@X = linkonce global {  } zeroinitializer               ; <{  }*> [#uses=0]

