// RUN: llvm-tblgen %s | grep WorldHelloCC | count 1
// XFAIL: vg_leak

class C<string n> {
  string name = n;
}

multiclass Names<string n, string m> {
   def CC : C<n>;
   def World#NAME#CC : C<m>;
}

defm Hello : Names<"hello", "world">;
