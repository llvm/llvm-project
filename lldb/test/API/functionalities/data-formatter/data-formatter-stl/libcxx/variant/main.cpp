#include <cstdio>
#include <string>
#include <vector>

// If we have libc++ 4.0 or greater we should have <variant>
// According to libc++ C++1z status page https://libcxx.llvm.org/cxx1z_status.html
#if _LIBCPP_VERSION >= 4000
#include <variant>
#define HAVE_VARIANT 1
#else
#define HAVE_VARIANT 0
#endif

struct S {
  operator int() { throw 42; }
} ;


int main()
{
    bool has_variant = HAVE_VARIANT ;

    printf( "%d\n", has_variant ) ; // break here

#if HAVE_VARIANT == 1
    std::variant<int, double, char> v1;
    std::variant<int, double, char> &v1_ref = v1;
    std::variant<int, double, char> v2;
    std::variant<int, double, char> v3;
    std::variant<std::variant<int,double,char>> v_v1 ;
    std::variant<int, double, char> v_no_value;
    // The next variant has 300 types, meaning the type index does not fit in
    // a byte and must be `unsigned short` instead of `unsigned char` when
    // using the unstable libc++ ABI. With stable libc++ ABI, the type index
    // is always just `unsigned int`.
    std::variant<
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int, int, int, int, int, int, int,
        int, int, int, int, int, int>
        v_300_types_no_value;

    v1 = 12; // v contains int
    v_v1 = v1 ;
    int i = std::get<int>(v1);
    printf( "%d\n", i ); // break here

    v2 = 2.0 ;
    double d = std::get<double>(v2) ;
    printf( "%f\n", d );

    v3 = 'A' ;
    char c = std::get<char>(v3) ;
    printf( "%d\n", c );

    // Checking v1 above and here to make sure we done maintain the incorrect
    // state when we change its value.
    v1 = 2.0;
    d = std::get<double>(v1) ;
    printf( "%f\n", d ); // break here

     try {
       v_no_value.emplace<0>(S());
     } catch( ... ) {}

     printf( "%zu\n", v_no_value.index() ) ;

     try {
       v_300_types_no_value.emplace<0>(S());
     } catch (...) {
     }

     printf("%zu\n", v_300_types_no_value.index());
#endif

    return 0; // break here
}
