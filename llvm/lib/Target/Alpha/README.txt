***

add gcc builtins for alpha instructions


***

custom expand byteswap into nifty 
extract/insert/mask byte/word/longword/quadword low/high
sequences

***

see if any of the extract/insert/mask operations can be added

***

match more interesting things for cmovlbc cmovlbs (move if low bit clear/set)

***

lower srem and urem

remq(i,j):  i - (j * divq(i,j)) if j != 0
remqu(i,j): i - (j * divqu(i,j)) if j != 0
reml(i,j):  i - (j * divl(i,j)) if j != 0
remlu(i,j): i - (j * divlu(i,j)) if j != 0

***

add crazy vector instructions (MVI):

(MIN|MAX)(U|S)(B8|W4) min and max, signed and unsigned, byte and word
PKWB, UNPKBW pack/unpack word to byte
PKLB UNPKBL pack/unpack long to byte
PERR pixel error (sum accross bytes of bytewise abs(i8v8 a - i8v8 b))

cmpbytes bytewise cmpeq of i8v8 a and i8v8 b (not part of MVI extentions)

this has some good examples for other operations that can be synthesised well 
from these rather meager vector ops (such as saturating add).
http://www.alphalinux.org/docs/MVI-full.html
