#ifndef LIB_H_IN
#define LIB_H_IN

template <typename T> class Wrapper {
public:
  [[gnu::abi_tag("test")]] Wrapper(){};

  [[gnu::abi_tag("test")]] ~Wrapper(){};
};

#endif // _H_IN

