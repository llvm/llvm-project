// Note: This test requires compiler support for DWARF entry values.

int dummy;
volatile int global = 0;

template <typename... T> __attribute__((optnone)) void use(T...) {}

struct S1 {
  int field1 = 123;
  int *field2 = &field1;
};

__attribute__((noinline)) void func1(int &sink) {
  // First use works around a compiler "bug" where an unused variable gets
  // no location descriptions. The second use overwrites the function arguments
  // with other values.
  use<int &>(sink);
  use<int &>(dummy);

  ++global;
  //% prefix = "FUNC1-GNU" if "GNU" in self.name else "FUNC1-V5"
  //% self.filecheck("image lookup -v -a $pc", "main.cpp", "-check-prefix="+prefix)
  // FUNC1-GNU: name = "sink", type = "int &", location = DW_OP_GNU_entry_value
  // FUNC1-V5: name = "sink", type = "int &", location = DW_OP_entry_value
}

__attribute__((noinline)) void func2(int &sink, int x) {
  use<int &, int>(sink, x);
  use<int &, int>(dummy, 0);

  ++global;
  //% self.filecheck("expr x", "main.cpp", "-check-prefix=FUNC2-EXPR1")
  //% self.filecheck("expr sink", "main.cpp", "-check-prefix=FUNC2-EXPR2")
  // FUNC2-EXPR1: ${{.*}} = 123
  // FUNC2-EXPR2: ${{.*}} = 2
}

__attribute__((noinline)) void func3(int &sink, int *p) {
  use<int &, int *>(sink, p);
  use<int &, int *>(dummy, nullptr);

  //% self.filecheck("expr *p", "main.cpp", "-check-prefix=FUNC3-EXPR")
  // FUNC3-EXPR: (int) ${{.*}} = 123
}

__attribute__((noinline)) void func4_amb(int &sink, int x) {
  use<int &, int>(sink, x);
  use<int &, int>(dummy, 0);

  ++global;
  //% self.filecheck("expr x", "main.cpp", "-check-prefix=FUNC4-EXPR-FAIL",
  //%     expect_cmd_failure=True)
  //% self.filecheck("expr sink", "main.cpp","-check-prefix=FUNC4-EXPR",
  //%     expect_cmd_failure=True)
  // FUNC4-EXPR-FAIL: couldn't get the value of variable x: Could not evaluate
  // DW_OP_entry_value. FUNC4-EXPR: couldn't get the value of variable sink:
  // Could not evaluate DW_OP_entry_value.
}

__attribute__((noinline)) void func5_amb() {}

__attribute__((noinline)) void func6(int &sink, int x) {
  if (sink > 0)
    func4_amb(sink, x); /* tail (taken) */
  else
    func5_amb(); /* tail */
}

__attribute__((noinline)) void func7(int &sink, int x) {
  //% self.filecheck("bt", "main.cpp", "-check-prefix=FUNC7-BT")
  // FUNC7-BT: func7
  // FUNC7-BT-NEXT: [inlined] func8_inlined
  // FUNC7-BT-NEXT: [inlined] func9_inlined
  // FUNC7-BT-NEXT: func10
  use<int &, int>(sink, x);
  use<int &, int>(dummy, 0);

  ++global;
  //% self.filecheck("expr x", "main.cpp", "-check-prefix=FUNC7-EXPR1")
  //% self.filecheck("expr sink", "main.cpp", "-check-prefix=FUNC7-EXPR2")
  // FUNC7-EXPR1: ${{.*}} = 123
  // FUNC7-EXPR2: ${{.*}} = 5
}

__attribute__((always_inline)) void func8_inlined(int &sink, int x) {
  func7(sink, x);
}

__attribute__((always_inline)) void func9_inlined(int &sink, int x) {
  func8_inlined(sink, x);
}

__attribute__((noinline, disable_tail_calls)) void func10(int &sink, int x) {
  func9_inlined(sink, x);
}

__attribute__((noinline)) void func11_tailcalled(int &sink, int x) {
  //% self.filecheck("bt", "main.cpp", "-check-prefix=FUNC11-BT")
  // FUNC11-BT: func11_tailcalled{{.*}}
  // FUNC11-BT-NEXT: func12{{.*}} [artificial]
  use<int &, int>(sink, x);
  use<int &, int>(dummy, 0);

  ++global;
  //% self.filecheck("expr x", "main.cpp", "-check-prefix=FUNC11-EXPR1")
  //% self.filecheck("expr sink", "main.cpp", "-check-prefix=FUNC11-EXPR2")
  // FUNC11-EXPR1: ${{.*}} = 123
  // FUNC11-EXPR2: ${{.*}} = 5
}

__attribute__((noinline)) void func12(int &sink, int x) {
  func11_tailcalled(sink, x);
}

__attribute__((noinline)) void func13(int &sink, int x) {
  //% self.filecheck("bt", "main.cpp", "-check-prefix=FUNC13-BT")
  // FUNC13-BT: func13{{.*}}
  // FUNC13-BT-NEXT: func14{{.*}}
  use<int &, int>(sink, x);
  use<int &, int>(dummy, 0);

  ++global;

  //% self.filecheck("expr x", "main.cpp", "-check-prefix=FUNC13-EXPR1")
  //% self.filecheck("expr sink", "main.cpp", "-check-prefix=FUNC13-EXPR2")
  // FUNC13-EXPR1: ${{.*}} = 123
  // FUNC13-EXPR2: ${{.*}} = 5
}

__attribute__((noinline, disable_tail_calls)) void
func14(int &sink, void (*target_no_tailcall)(int &, int)) {
  // Move the call target into a register that won't get clobbered. Do this
  // by calling the same indirect target twice, and hoping that regalloc is
  // 'smart' enough to stash the call target in a non-clobbered register.
  //
  // llvm.org/PR43926 tracks work in the compiler to emit call targets which
  // describe non-clobbered values.
  target_no_tailcall(sink, 123);
  target_no_tailcall(sink, 123);
}

/// A structure that is guaranteed -- when passed to a callee by value -- to be
/// passed via a pointer to a temporary copy in the caller. On x86_64 & aarch64
/// only.
struct StructPassedViaPointerToTemporaryCopy {
  // Under the 64-bit AAPCS, a struct larger than 16 bytes is not SROA'd, and
  // is instead passed via pointer to a temporary copy.
  long a, b, c;
  StructPassedViaPointerToTemporaryCopy() : a(1), b(2), c(3) {}

  // Failing that, a virtual method forces passing via pointer to a temporary
  // copy under the common calling conventions (e.g. 32/64-bit x86, Linux/Win,
  // according to https://www.agner.org/optimize/calling_conventions.pdf).
  virtual void add_vtable() {}
};

__attribute__((noinline)) void func15(StructPassedViaPointerToTemporaryCopy S) {
  use<StructPassedViaPointerToTemporaryCopy &>(S);
  use<int &>(dummy);

  ++global;
  //% self.filecheck("expr S", "main.cpp", "-check-prefix=FUNC15-EXPR")
  // FUNC15-EXPR: (a = 1, b = 2, c = 3)
}

__attribute__((disable_tail_calls)) int main() {
  int sink = 0;
  S1 s1;

  // Test location dumping for DW_OP_entry_value.
  func1(sink);

  sink = 2;
  // Test evaluation of "DW_OP_constu" in the parent frame.
  func2(sink, 123);

  // Test evaluation of "DW_OP_fbreg -24, DW_OP_deref" in the parent frame.
  // Disabled for now, see: llvm.org/PR43343
#if 0
  func3(sink, s1.field2);
#endif

  // The sequences `main -> func4 -> func{5,6}_amb -> sink` are both plausible.
  // Test that lldb doesn't attempt to guess which one occurred: entry value
  // evaluation should fail.
  func6(sink, 123);

  sink = 5;
  // Test that evaluation can "see through" inlining.
  func10(sink, 123);

  // Test that evaluation can "see through" tail calls.
  func12(sink, 123);

  // Test that evaluation can "see through" an indirect tail call.
  func14(sink, func13);

  // Test evaluation of an entry value that dereferences a temporary stack
  // slot set up by the caller for a StructPassedViaPointerToTemporaryCopy.
  func15(StructPassedViaPointerToTemporaryCopy());

  return 0;
}
