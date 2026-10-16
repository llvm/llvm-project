extern "C" {
int linkage_library_anchor() { return 0; }
int linkage_target() { return 42; }
int linkage_global = 73;
int present_weak() { return 29; }

// A C++ expression declaration intentionally omits extern "C". The existing
// alternate-mangling and basename fallback must still find this C definition.
int cpp_fallback_target() { return 17; }
}
