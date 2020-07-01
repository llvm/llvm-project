struct One {
  int one = 142;
  constexpr One() = default;
  virtual ~One();
};

struct Two : One {
  int two = 242;
  constexpr Two() = default;
  ~Two() override;
};
