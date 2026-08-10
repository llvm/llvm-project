struct NoCopyCstr {
  NoCopyCstr() {}
  // No copy constructor but a move constructor means we have an
  // implicitly deleted copy constructor (C++11 [class.copy]p7, p18).
  NoCopyCstr(NoCopyCstr &&);
};
struct IndirectlyDeletedCopyCstr {
  // This field indirectly deletes the implicit copy constructor.
  NoCopyCstr field;
  // Completing in the constructor or constructing the class
  // will cause Sema to declare the special members of IndirectlyDeletedCopyCstr.
  // If we correctly set the deleted implicit copy constructor in NoCopyCstr then this
  // should have propagated to this record and Clang won't crash.
  IndirectlyDeletedCopyCstr() { //%self.expect_expr("IndirectlyDeletedCopyCstr x; 1+1", result_type="int", result_value="2")
                                //%self.dbg.GetCommandInterpreter().HandleCompletion("e ", len("e "), 0, -1, lldb.SBStringList())
  }
};
int main() {
  IndirectlyDeletedCopyCstr{};
}
