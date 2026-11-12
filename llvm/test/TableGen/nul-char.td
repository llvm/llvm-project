// RUN: cat %s | tr '@' '\0' > %t
// RUN: not llvm-tblgen -DERROR1 %t 2>&1 | FileCheck --check-prefix=ERROR1 %s

// This test file checks that NUL is treated as an invalid character.
// Each at sign is replaced with a NUL before running the test.

#ifdef ERROR1

// ERROR1: error: NUL character is invalid in source; treated as space
// ERROR1: error: NUL character is invalid in source; treated as space
// ERROR1: error: NUL character is invalid in source; treated as space
// ERROR1: error: NUL character is invalid in source; treated as space
// ERROR1: error: expected ';' after declaration

def Foo@ {
  int @ ID = 42;
}

@

// Comment with a NUL @ there. They are ignored in comments.

def Bar {
  int Biggie = 12345@789;
}

#endif

