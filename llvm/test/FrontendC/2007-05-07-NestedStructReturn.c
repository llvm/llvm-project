// RUN: %llvmgcc %s -S -fnested-functions -o - | grep {sret *%agg.result}

struct X { long m, n, o, p; };

struct X p(int n) {
  struct X c(int m) {
    struct X x;
    x.m = m;
    x.n = n;
    return x;
  }
  return c(n);
}
