for (int c0 = 1; c0 < n - 1; c0 += 2)
  for (int c1 = c0 - 1; c1 <= c0; c1 += 1)
    S(c1);
if (n >= 1)
  for (int c1 = n - 2; c1 < n; c1 += 1)
    if (2 * ((c1 + 2) / 2) >= n)
      S(c1);
