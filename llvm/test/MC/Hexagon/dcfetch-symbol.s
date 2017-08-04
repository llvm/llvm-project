# RUN: not llvm-mc -arch=hexagon -filetype=obj %s

#CHECK: 9400c000 { dcfetch(r0 + #0) }

junk:
{
  dcfetch(r0 + #junk)
}
