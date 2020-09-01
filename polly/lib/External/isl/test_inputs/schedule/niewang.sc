# In earlier version of isl, this test case would take an inordinate
# amount of time (in the order of 30s versus 0.1s in later versions)
# due to a call to isl_basic_set_coefficients.
# Check that this no longer happens in the sense that this test case
# would stand out if it were to take that long again.
# The actual schedule that is produced is not that important,
# but it is different depending on whether the whole-component scheduler
# is being used, so pick a particular setting.
# OPTIONS: --no-schedule-whole-component
domain: { S_3[0:223, 0:223, 0:15, 0:3, 0:6, 0:6, 0:15]; group0[0:3,
0:223, 0:223, 0:15] }
validity: { group0[i0 = 0:3, i1 = 0:223, i2 = 0:223, i3 = 0:15]
-> S_3[h = 0:223, w = 0:223, c0 = 0:15, kc1 = i0, kh = 3 + i1 - h,
kw = 3 + i2 - w, kc0 = i3] : -3 + i1 <= h <= 3 + i1 and -3 + i2 <=
w <= 3 + i2; group0[i0 = 0:3, i1 = 0:223, i2 = 0:223, i3 = 0:15] ->
group0[i0, i1, i2, i3] : (i1) mod 2 = 0 and (i2) mod 2 = 0; S_3[h =
0:223, w = 0:223, c0 = 0:15, kc1 = 0:3, kh = 0:6, kw = 0:6, kc0 = 0:14]
-> S_3[h' = h, w' = w, c0' = c0, kc1' = kc1, kh' = kh, kw' = kw, kc0'
= 1 + kc0]; S_3[h = 0:223, w = 0:223, c0 = 0:15, kc1 = 0:3, kh = 0:6,
kw = 0:5, kc0 = 15] -> S_3[h' = h, w' = w, c0' = c0, kc1' = kc1, kh'
= kh, kw' = 1 + kw, kc0' = 0]; S_3[h = 0:223, w = 0:223, c0 = 0:15,
kc1 = 0:3, kh = 0:5, kw = 6, kc0 = 15] -> S_3[h' = h, w' = w, c0' =
c0, kc1' = kc1, kh' = 1 + kh, kw' = 0, kc0' = 0]; S_3[h = 0:223, w =
0:223, c0 = 0:15, kc1 = 0:2, kh = 6, kw = 6, kc0 = 15] -> S_3[h' =
h, w' = w, c0' = c0, kc1' = 1 + kc1, kh' = 0, kw' = 0, kc0' = 0] }
proximity: { group0[i0 = 0:3, i1 = 0:223, i2 = 0:223, i3 = 0:15] ->
S_3[h = 0:223, w = 0:223, c0 = 0:15, kc1 = i0, kh = 3 + i1 - h, kw =
3 + i2 - w, kc0 = i3] : -3 + i1 <= h <= 3 + i1 and -3 + i2 <= w <=
3 + i2 }
