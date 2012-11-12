// RUN: clang -fsyntax-only -Xclang -verify %s

int f0(void) {} // expected-warning {{control reaches end of non-void function}}

