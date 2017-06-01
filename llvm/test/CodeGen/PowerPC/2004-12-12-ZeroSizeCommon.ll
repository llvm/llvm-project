; RUN: llc -verify-machineinstrs < %s -march=ppc32 | not grep .comm.*X,0

@X = linkonce global {  } zeroinitializer               ; <{  }*> [#uses=0]

