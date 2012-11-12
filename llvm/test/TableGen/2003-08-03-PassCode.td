// RUN: llvm-tblgen %s
// XFAIL: vg_leak

class test<code C> {
  code Code = C;
}

def foo : test<[{ hello world! }]>;
