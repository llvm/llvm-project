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

namespace member {
struct One {
  int member = 147;
  constexpr One() = default;
  virtual ~One();
};

struct Two {
  One one;
  int member = 247;
  constexpr Two() = default;
  virtual ~Two();
};
} // namespace member

namespace array {
struct One {
  int member = 174;
  constexpr One() = default;
  virtual ~One();
};

struct Two {
  One one[3];
  int member = 274;
  constexpr Two() = default;
  virtual ~Two();
};
} // namespace array

namespace result {
struct One {
  int member;
  One(int member);
  virtual ~One();
};

struct Two {
  int member;
  Two(int member);
  One one() const;
  virtual ~Two();
};
} // namespace result

namespace func_shadow {
void One(int);
struct One {
  int one = 142;
  constexpr One() = default;
  virtual ~One();
};
void One(float);
} // namespace func_shadow
