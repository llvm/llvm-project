template <class T> class FOO {
public:
  FOO() : t(0) {}

  T DoIt(T ti);

private:
  T t;
};

template <class T> T FOO<T>::DoIt(T ti) { // HEADER:  2| [[@LINE]]|template
  for (T I = 0; I < ti; I++) {            // HEADER: 22| [[@LINE]]|  for (T
    t += I;                               // HEADER: 20| [[@LINE]]|    t += I;
    if (I > ti / 2)                       // HEADER: 20| [[@LINE]]|    if (I > ti 
      t -= 1;                             // HEADER:  8| [[@LINE]]|      t -= 1;
  }                                       // HEADER: 20| [[@LINE]]|  }
                                          // HEADER:  2| [[@LINE]]|
  return t;                               // HEADER:  2| [[@LINE]]|  return t;
}

// To generate the binaries which correspond to this file, you must first
// compile a program with two calls to Foo<int>::DoIt(10) for each desired
// architecture. Collect a raw profile from any one of these binaries, index
// it, and check it in along with the executables.
