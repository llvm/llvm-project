// RUN: llvm-tblgen --no-warn-on-unused-template-args %s | FileCheck %s
// XFAIL: vg_leak

class Or4<bits<8> Val> {
  bits<8> V = {Val{7}, Val{6}, Val{5}, Val{4}, Val{3}, 1, Val{1}, Val{0} };
}

class Whatev<bits<8> x>;

class Whatever<bits<8> x> {
  bits<8> W = {x{0}, x{1}, x{2}, x{3}, x{4}, x{5}, x{6}, x{7} };
}

multiclass X<bits<8> BaseOpc> {
 def bar : Whatev<Or4<BaseOpc>.V >;
}

multiclass Y<bits<8> BaseOpc> {
 def foo : Whatever<Or4<BaseOpc>.V >;
}

defm a : X<4>;

// CHECK: def abar

defm b : Y<8>;

// CHECK: def bfoo
// CHECK-NEXT: bits<8> W = { 0, 0, 1, 1, 0, 0, 0, 0 };
