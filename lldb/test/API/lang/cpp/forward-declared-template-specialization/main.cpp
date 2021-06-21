// Forward declare a template and a specialization;
template <typename T> class Temp;
template <> class Temp<int>;

// Force that debug informatin for the specialization is emitted.
// Clang and GCC will create debug information that lacks any description
// of the template argument 'int'.
Temp<int> *a;

// Define the template and create an implicit instantiation.
template <typename T> class Temp { int f; };
Temp<float> b;

int main() {
  return 0; // break here
}
