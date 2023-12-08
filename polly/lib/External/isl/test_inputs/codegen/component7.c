S();
for (int c0 = 0; c0 < K; c0 += 32)
  for (int c1 = c0; c1 <= min(K - 1, c0 + 31); c1 += 1)
    T(c1);
