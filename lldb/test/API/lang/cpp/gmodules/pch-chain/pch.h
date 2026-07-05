#ifndef PCH_H_IN
#define PCH_H_IN

static const int kAlignment = 64;

struct [[gnu::aligned(kAlignment)]] Submatrix {
  struct RowCol origin;
  struct RowCol size;
};

struct [[gnu::aligned(kAlignment)]] MatrixData {
  struct Submatrix section;
  unsigned stride;
};

#endif // _H_IN
