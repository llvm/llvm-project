; RUN: llvm-as < %s | opt -anders-aa -disable-output

define void @x(i16 %Y) {
entry:
  %tmp = call i16 asm "bswap $0", "=r,r"(i16 %Y)
  ret void
}

