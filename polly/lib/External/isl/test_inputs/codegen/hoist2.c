for (int c0 = 1; c0 <= min(5, -t1 + 64 * b + 9); c0 += 1)
  if (c0 <= 3 || b == 1)
    for (int c1 = t1 - 64 * b + 64; c1 <= min(70, -c0 + 73); c1 += 64)
      if (c0 <= 3 || c1 == t1)
        A(c0, 64 * b + c1 - 8);
