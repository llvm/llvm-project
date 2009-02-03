// RUN: %llvmgcc -xc++ -S -o - %s | not grep weak
// The template should compile to linkonce linkage, not weak linkage.

template<class T>
void thefunc();

template<class T>
inline void thefunc() {}

void test() {
  thefunc<int>();
}

