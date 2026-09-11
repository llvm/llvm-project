// The "v1" variant of the library. See hidden/v.cpp for the "v2" variant that
// it gets replaced with: same soname, different content and different UUID.

extern "C" int only_in_v1() { return 101; }

__thread int tls_var = 701;

extern "C" int get_tls_var() { return tls_var; }

extern "C" int common_func() { return 1; }
