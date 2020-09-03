if (d <= -1) {
  for (int c0 = 0; c0 < -(-d % 2) - d + w; c0 += 1) {
    A(c0);
    B(c0);
  }
} else {
  for (int c0 = 0; c0 < (d % 2) - d + w; c0 += 1)
    A(c0);
}
