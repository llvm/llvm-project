struct Opaque {
  int i, j, k;
} *global;
struct Opaque *getOpaque() { return &global; }
