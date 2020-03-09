// Note: This test requires the SysV AMD64 ABI to be in use, and requires
// compiler support for DWARF entry values.

// Inhibit dead-arg-elim by using 'x'.
template<typename T> __attribute__((noinline)) void use(T x) {
  asm volatile (""
      /* Outputs */  :
      /* Inputs */   : "g"(x)
      /* Clobbers */ :
  );
}

// Destroy %rsi in the current frame.
#define DESTROY_RSI \
  asm volatile ("xorq %%rsi, %%rsi" \
      /* Outputs */  : \
      /* Inputs */   : \
      /* Clobbers */ : "rsi" \
  );

// Destroy %rbx in the current frame.
#define DESTROY_RBX \
  asm volatile ("xorq %%rbx, %%rbx" \
      /* Outputs */  : \
      /* Inputs */   : \
      /* Clobbers */ : "rbx" \
  );

struct S1 {
  int field1 = 123;
  int *field2 = &field1;
};

__attribute__((noinline))
void func1(int &sink, int x) {
  use(x);

  // Destroy 'x' in the current frame.
  DESTROY_RSI;

  // NOTE: Currently, we do not generate DW_OP_entry_value for the 'x',
  // since it gets copied into a register that is not callee saved,
  // and we can not guarantee that its value has not changed.

  ++sink;

  // Destroy 'sink' in the current frame.
  DESTROY_RBX;

  //% self.filecheck("image lookup -va $pc", "main.cpp", "-check-prefix=FUNC1-DESC")
  // FUNC1-DESC: name = "sink", type = "int &", location = DW_OP_entry_value(DW_OP_reg5 RDI)
}

__attribute__((noinline))
void func2(int &sink, int x) {
  use(x);

  // Destroy 'x' in the current frame.
  DESTROY_RSI;

  //% self.filecheck("expr x", "main.cpp", "-check-prefix=FUNC2-EXPR-FAIL", expect_cmd_failure=True)
  // FUNC2-EXPR-FAIL: couldn't get the value of variable x: variable not available

  ++sink;

  // Destroy 'sink' in the current frame.
  DESTROY_RBX;

  //% self.filecheck("expr sink", "main.cpp", "-check-prefix=FUNC2-EXPR")
  // FUNC2-EXPR: ${{.*}} = 2
}

__attribute__((noinline))
void func3(int &sink, int *p) {
  use(p);

  // Destroy 'p' in the current frame.
  DESTROY_RSI;

  //% self.filecheck("expr *p", "main.cpp", "-check-prefix=FUNC3-EXPR")
  // FUNC3-EXPR: (int) ${{.*}} = 123

  ++sink;
}

__attribute__((noinline))
void func4_amb(int &sink, int x) {
  use(x);

  // Destroy 'x' in the current frame.
  DESTROY_RSI;

  //% self.filecheck("expr x", "main.cpp", "-check-prefix=FUNC4-EXPR-FAIL", expect_cmd_failure=True)
  // FUNC4-EXPR-FAIL: couldn't get the value of variable x: variable not available

  ++sink;

  // Destroy 'sink' in the current frame.
  DESTROY_RBX;

  //% self.filecheck("expr sink", "main.cpp", "-check-prefix=FUNC4-EXPR", expect_cmd_failure=True)
  // FUNC4-EXPR: couldn't get the value of variable sink: Could not evaluate DW_OP_entry_value.
}

__attribute__((noinline))
void func5_amb() {}

__attribute__((noinline))
void func6(int &sink, int x) {
  if (sink > 0)
    func4_amb(sink, x); /* tail (taken) */
  else
    func5_amb(); /* tail */
}

__attribute__((noinline))
void func7(int &sink, int x) {
  //% self.filecheck("bt", "main.cpp", "-check-prefix=FUNC7-BT")
  // FUNC7-BT: func7
  // FUNC7-BT-NEXT: [inlined] func8_inlined
  // FUNC7-BT-NEXT: [inlined] func9_inlined
  // FUNC7-BT-NEXT: func10
  use(x);

  // Destroy 'x' in the current frame.
  DESTROY_RSI;

  //% self.filecheck("expr x", "main.cpp", "-check-prefix=FUNC7-EXPR-FAIL", expect_cmd_failure=True)
  // FUNC7-EXPR-FAIL: couldn't get the value of variable x: variable not available

  ++sink;

  // Destroy 'sink' in the current frame.
  DESTROY_RBX;

  //% self.filecheck("expr sink", "main.cpp", "-check-prefix=FUNC7-EXPR")
  // FUNC7-EXPR: ${{.*}} = 4
}

__attribute__((always_inline))
void func8_inlined(int &sink, int x) {
  func7(sink, x);
}

__attribute__((always_inline))
void func9_inlined(int &sink, int x) {
  func8_inlined(sink, x);
}

__attribute__((noinline, disable_tail_calls))
void func10(int &sink, int x) {
  func9_inlined(sink, x);
}

__attribute__((noinline))
void func11_tailcalled(int &sink, int x) {
  //% self.filecheck("bt", "main.cpp", "-check-prefix=FUNC11-BT")
  // FUNC11-BT: func11_tailcalled{{.*}}
  // FUNC11-BT-NEXT: func12{{.*}} [artificial]
  use(x);

  // Destroy 'x' in the current frame.
  DESTROY_RSI;

  //% self.filecheck("expr x", "main.cpp", "-check-prefix=FUNC11-EXPR-FAIL", expect_cmd_failure=True)
  // FUNC11-EXPR-FAIL: couldn't get the value of variable x: variable not available

  ++sink;

  // Destroy 'sink' in the current frame.
  DESTROY_RBX;

  //% self.filecheck("expr sink", "main.cpp", "-check-prefix=FUNC11-EXPR")
  // FUNC11-EXPR: ${{.*}} = 5
}

__attribute__((noinline))
void func12(int &sink, int x) {
  func11_tailcalled(sink, x);
}

__attribute__((noinline))
void func13(int &sink, int x) {
  //% self.filecheck("bt", "main.cpp", "-check-prefix=FUNC13-BT")
  // FUNC13-BT: func13{{.*}}
  // FUNC13-BT-NEXT: func14{{.*}}
  use(x);

  // Destroy 'x' in the current frame.
  DESTROY_RSI;

  //% self.filecheck("expr x", "main.cpp", "-check-prefix=FUNC13-EXPR-FAIL", expect_cmd_failure=True)
  // FUNC13-EXPR-FAIL: couldn't get the value of variable x: variable not available

  use(sink);

  // Destroy 'sink' in the current frame.
  DESTROY_RBX;

  //% self.filecheck("expr sink", "main.cpp", "-check-prefix=FUNC13-EXPR")
  // FUNC13-EXPR: ${{.*}} = 5
}

__attribute__((noinline, disable_tail_calls))
void func14(int &sink, void (*target_no_tailcall)(int &, int)) {
  // Move the call target into a register that won't get clobbered. Do this
  // by calling the same indirect target twice, and hoping that regalloc is
  // 'smart' enough to stash the call target in a non-clobbered register.
  //
  // llvm.org/PR43926 tracks work in the compiler to emit call targets which
  // describe non-clobbered values.
  target_no_tailcall(sink, 123);
  target_no_tailcall(sink, 123);
}

__attribute__((disable_tail_calls))
int main() {
  int sink = 0;
  S1 s1;

  // Test location dumping for DW_OP_entry_value.
  func1(sink, 123);

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

  // Test that evaluation can "see through" inlining.
  func10(sink, 123);

  // Test that evaluation can "see through" tail calls.
  func12(sink, 123);

  // Test that evaluation can "see through" an indirect tail call.
  func14(sink, func13);

  return 0;
}
