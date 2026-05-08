if (c1 >= d1 + 1) {
  for (int c0 = a2; c0 <= b2; c0 += 1)
    for (int c1_0 = c2; c1_0 <= d2; c1_0 += 1)
      s1(c0, c1_0);
} else {
  for (int c0 = a2; c0 <= min(a1 - 1, b2); c0 += 1)
    for (int c1_0 = c2; c1_0 <= d2; c1_0 += 1)
      s1(c0, c1_0);
  for (int c0 = a1; c0 <= b1; c0 += 1) {
    if (c0 >= a2 && b2 >= c0)
      for (int c1_0 = c2; c1_0 <= min(c1 - 1, d2); c1_0 += 1)
        s1(c0, c1_0);
    for (int c1_0 = c1; c1_0 <= d1; c1_0 += 1) {
      s0(c0, c1_0);
      if (c0 >= a2 && b2 >= c0 && c1_0 >= c2 && d2 >= c1_0)
        s1(c0, c1_0);
    }
    if (c0 >= a2 && b2 >= c0)
      for (int c1_0 = max(d1 + 1, c2); c1_0 <= d2; c1_0 += 1)
        s1(c0, c1_0);
  }
  for (int c0 = max(max(a1, b1 + 1), a2); c0 <= b2; c0 += 1)
    for (int c1_0 = c2; c1_0 <= d2; c1_0 += 1)
      s1(c0, c1_0);
}
