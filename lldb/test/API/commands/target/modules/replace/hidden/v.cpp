// The "v2" variant of the library. Built with the same soname as ../v.cpp so
// that it can stand in for it, but with different content so that the two are
// distinguishable by UUID and by the symbols they define.

extern "C" int only_in_v2() { return 202; }

__thread int tls_var = 702;

extern "C" int get_tls_var() { return tls_var; }

extern "C" int common_func() { return 2; }
