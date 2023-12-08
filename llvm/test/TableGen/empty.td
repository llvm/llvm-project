// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

defvar LongList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
defvar EmptyStr = "";

// Test !empty(dag).

// CHECK: def Drec1 {
// CHECK:   bit Empty = 1;
// CHECK:   bit NotEmpty = 0;
// CHECK: def Drec2 {
// CHECK:   bit Empty = 0;
// CHECK:   bit NotEmpty = 1;
// CHECK: def Drec3 {
// CHECK:   bit Empty = 0;
// CHECK:   bit NotEmpty = 1;

class D<dag ADag> {
  bit Empty = !empty(ADag);
  bit NotEmpty = !not(!empty(ADag));
}

def op;

def Drec1 : D<(op)>;
def Drec2 : D<(op "string")>;
def Drec3 : D<(op "string", 42)>;

// Test !empty(list).

// CHECK: def Lrec1 {
// CHECK:   bit Empty = 1;
// CHECK:   bit NotEmpty = 0;
// CHECK: def Lrec2 {
// CHECK:   bit Empty = 0;
// CHECK:   bit NotEmpty = 1;
// CHECK: def Lrec3 {
// CHECK:   bit Empty = 0;
// CHECK:   bit NotEmpty = 1;

class L<list<int> Ints> {
  bit Empty = !empty(Ints);
  bit NotEmpty = !not(!empty(Ints));
}

def Lrec1 : L<[]>;
def Lrec2 : L<[1]>;
def Lrec3 : L<LongList>;

// Test !empty(string).

// CHECK: def Srec1 {
// CHECK:   bit Empty = 1;
// CHECK:   bit NotEmpty = 0;
// CHECK: def Srec2 {
// CHECK:   bit Empty = 0;
// CHECK:   bit NotEmpty = 1;
// CHECK: def Srec3 {
// CHECK:   bit Empty = 0;
// CHECK:   bit NotEmpty = 1;

class S<string Str> {
  bit Empty = !empty(Str);
  bit NotEmpty = !not(!empty(Str));
}

def Srec1 : S<EmptyStr>;
def Srec2 : S<"a">;
def Srec3 : S<"ab">;

