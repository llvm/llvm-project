template <class T> class FOO {
public:
  FOO() : t(0) {}

  T DoIt(T ti);

private:
  T t;
};

template <class T> T FOO<T>::DoIt(T ti) { // HEADER: [[@LINE]]|  2|template
  for (T I = 0; I < ti; I++) {            // HEADER: [[@LINE]]| 22|  for (T
    t += I;                               // HEADER: [[@LINE]]| 20|   t += I;
    if (I > ti / 2)                       // HEADER: [[@LINE]]| 20|   if (I > ti
      t -= 1;                             // HEADER: [[@LINE]]| 20|     t -= 1;
  }                                       // HEADER: [[@LINE]]| 20| }
                                          // HEADER: [[@LINE]]|  2|
  return t;                               // HEADER: [[@LINE]]|  2|  return t;
}

// To generate the binaries which correspond to this file, you must first
// compile a program with two calls to Foo<int>::DoIt(10) for each desired
// architecture. Collect a raw profile from any one of these binaries, index
// it, and check it in along with the executables.

// Note:
// The line execution count on line 14 is incorrect because of a bug in the
// checked-in coverage mappings, which has subsequently been fixed in clang.
