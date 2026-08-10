void f1() {
}

inline __attribute__((always_inline)) void f2() {
  f1();
}

int main() {
  f2();
}

// Build instructions:
// 1) clang++ -### -gsplit-dwarf split-dwarf-test.cc -o split-dwarf-test
// 2) Replace the value "-fdebug-compilation-dir" flag to "Output"
//      (this is the temp directory used by lit).
// 3) Manually run clang-cc1, objcopy and ld invocations.
// 4) Copy the binary and .dwo file to the Inputs directory. Make sure the
//    .dwo file will be available for symbolizer (use test RUN-lines to copy
//    the .dwo file to a directory
//    <execution_directory>/<directory_provided_in_fdebug_compilation_dir>.
