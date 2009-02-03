; RUN: llvm-as < %s | opt -anders-aa -disable-output

define void @foo() { ret void }
