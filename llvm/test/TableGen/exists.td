// RUN: llvm-tblgen --no-warn-on-unused-template-args %s | FileCheck %s
// XFAIL: vg_leak

class A;
def a0 : A;

class A_check<string name>{
  int exists = !exists<A>(name);
}

def a0_exists : A_check<"a0">;
def a1_missing : A_check<"a1">;


// Subclasses are allowed.

class B;
class SubOfB : B;
class B_check<string name> {
  int exists = !exists<B>(name);
}

def sub : SubOfB;

def sub_exists : B_check<"sub">;
def a0_is_not_sub_of_B : B_check<"a0">;


// Self-references are allowed.

class Self_check<string name> {
  int exists = !exists<Self_check>(name);
}

def self_reference : Self_check<"self_reference">; // Self-reference
// There is no record called `current` in current context though we will define it below.
def current_missing : Self_check<"current">;
def current : Self_check<"current">;

// CHECK: def a0_exists {
// CHECK:   int exists = 1;
// CHECK: }

// CHECK: def a0_is_not_sub_of_B {
// CHECK:   int exists = 0;
// CHECK: }

// CHECK: def a1_missing {
// CHECK:   int exists = 0;
// CHECK: }

// CHECK: def current {
// CHECK:   int exists = 1;
// CHECK: }

// `current` doesn't exist because we define it below `current_missing`.
// CHECK: def current_missing {
// CHECK:   int exists = 0;
// CHECK: }

// CHECK: def self_reference {
// CHECK:   int exists = 1;
// CHECK: }

// CHECK: def sub_exists {
// CHECK:   int exists = 1;
// CHECK: }
