// RUN: %llvmgxx %s -S -o /dev/null

// This is PR421

struct Strongbad {
    Strongbad(const char *str );
    ~Strongbad();
    operator const char *() const;
};

void TheCheat () {
  Strongbad foo(0);
  Strongbad dirs[] = { Strongbad(0) + 1};
}
