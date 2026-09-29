; RUN: llc < %s

define void @main()  {
if.end:
  br label %block.i.i

block.i.i:
  %tmpbb = load i8, ptr undef
  %tmp54 = zext i8 %tmpbb to i64
  %tmp59 = and i64 %tmp54, 8
  %tmp60 = add i64 %tmp59, 3691045929300498764
  %tmp62 = sub i64 %tmp60, 3456506383779105993
  %tmp63 = xor i64 1050774804270620004, %tmp62
  %tmp65 = xor i64 %tmp62, 234539545521392771
  %tmp67 = or i64 %tmp65, %tmp63
  %tmp71 = xor i64 %tmp67, 6781485823212740913
  %tmp72 = trunc i64 %tmp71 to i32
  %tmp74 = lshr i32 2, %tmp72
  store i32 %tmp74, ptr undef
  br label %block.i.i
}
