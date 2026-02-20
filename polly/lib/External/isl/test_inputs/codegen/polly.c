if (p_0 >= p_1 + 1) {
  for (int c0 = 0; c0 <= p_1 - 4 * floord(p_1 + 2, 4); c0 += 1)
    Stmt2(c0);
  if (4 * floord(p_1 + 2, 4) >= p_1 + 1)
    for (int c0 = 0; c0 <= p_1 - 4 * floord(p_1 + 2, 4) + 4; c0 += 1)
      Stmt2(c0);
} else if (p_0 >= p_1 + 4 * floord(p_0 - p_1, 4) + 1) {
  for (int c0 = 0; c0 <= p_0 - 4 * floord(p_0 + 2, 4); c0 += 1)
    Stmt2(c0);
  if (4 * floord(p_0 + 2, 4) >= p_0 + 1)
    for (int c0 = 0; c0 <= p_0 - 4 * floord(p_0 + 2, 4) + 4; c0 += 1)
      Stmt2(c0);
} else if (4 * floord(p_0 + 2, 4) >= p_0 + 1) {
  for (int c0 = 0; c0 <= p_0 - 4 * floord(p_0 + 2, 4) + 4; c0 += 1)
    Stmt2(c0);
} else {
  for (int c0 = 0; c0 <= p_0 - 4 * floord(p_0 + 2, 4); c0 += 1)
    Stmt2(c0);
}
