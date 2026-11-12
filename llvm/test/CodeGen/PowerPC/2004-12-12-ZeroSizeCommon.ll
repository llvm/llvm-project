; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | not grep .comm.*X,0

@X = linkonce global {  } zeroinitializer               ; <ptr> [#uses=0]

