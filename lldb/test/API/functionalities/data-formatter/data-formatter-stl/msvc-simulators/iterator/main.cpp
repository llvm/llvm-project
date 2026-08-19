// Layout approximation of MSVC STL vector iterators.

namespace std {
template <class T> class _Vector_iterator {
public:
  explicit _Vector_iterator(T *ptr) : _Ptr(ptr) {}
  T *_Ptr;
};

template <class T> class _Vector_const_iterator {
public:
  explicit _Vector_const_iterator(const T *ptr) : _Ptr(ptr) {}
  const T *_Ptr;
};
} // namespace std

int main() {
  int item = 3;
  std::_Vector_iterator<int> it(&item);
  std::_Vector_const_iterator<int> cit(&item);
  return 0; // break here
}
