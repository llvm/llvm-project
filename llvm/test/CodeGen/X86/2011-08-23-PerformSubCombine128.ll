; RUN: llc -mtriple=x86_64-- -O2 < %s

define void @test(i64 %add127.tr.i2686) {
entry:
  %conv143.i2687 = and i64 %add127.tr.i2686, 72057594037927935
  %conv76.i2623 = zext i64 %conv143.i2687 to i128
  %mul148.i2338 = mul i128 0, %conv76.i2623
  %add149.i2339 = add i128 %mul148.i2338, 0
  %add.i2303 = add i128 0, 170141183460469229370468033484042534912
  %add6.i2270 = add i128 %add.i2303, 0
  %sub58.i2271 = sub i128 %add6.i2270, %add149.i2339
  %add71.i2272 = add i128 %sub58.i2271, 0
  %add105.i2273 = add i128 %add71.i2272, 0
  %add116.i2274 = add i128 %add105.i2273, 0
  %shr124.i2277 = lshr i128 %add116.i2274, 56
  %add116.tr.i2280 = trunc i128 %add116.i2274 to i64
  ret void
}
