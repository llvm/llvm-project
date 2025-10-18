// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s

// CHECK: ------------- Classes -----------------
// CHECK-NEXT: class X<int X:a = ?, int X:b = ?> {
// CHECK-NEXT:   list<int> x = !listsplat(X:a, X:b);
// CHECK-NEXT: }
// CHECK-NEXT: class Y<string Y:a = ?, int Y:b = ?> {
// CHECK-NEXT:   list<string> x = !listsplat(Y:a, Y:b);
// CHECK-NEXT: }
// CHECK-NEXT: ------------- Defs -----------------
// CHECK-NEXT: def DX00 {      // X
// CHECK-NEXT:   list<int> x = [];
// CHECK-NEXT: }
// CHECK-NEXT: def DX01 {      // X
// CHECK-NEXT:   list<int> x = [0];
// CHECK-NEXT: }
// CHECK-NEXT: def DX02 {      // X
// CHECK-NEXT:   list<int> x = [0, 0];
// CHECK-NEXT: }
// CHECK-NEXT: def DX10 {      // X
// CHECK-NEXT:   list<int> x = [];
// CHECK-NEXT: }
// CHECK-NEXT: def DX11 {      // X
// CHECK-NEXT:   list<int> x = [1];
// CHECK-NEXT: }
// CHECK-NEXT: def DX12 {      // X
// CHECK-NEXT:   list<int> x = [1, 1];
// CHECK-NEXT: }
// CHECK-NEXT: def DYa0 {      // Y
// CHECK-NEXT:   list<string> x = [];
// CHECK-NEXT: }
// CHECK-NEXT: def DYa1 {      // Y
// CHECK-NEXT:   list<string> x = ["a"];
// CHECK-NEXT: }
// CHECK-NEXT: def DYa2 {      // Y
// CHECK-NEXT:   list<string> x = ["a", "a"];
// CHECK-NEXT: }
// CHECK-NEXT: def DYe0 {      // Y
// CHECK-NEXT:   list<string> x = [];
// CHECK-NEXT: }
// CHECK-NEXT: def DYe1 {      // Y
// CHECK-NEXT:   list<string> x = [""];
// CHECK-NEXT: }
// CHECK-NEXT: def DYe2 {      // Y
// CHECK-NEXT:   list<string> x = ["", ""];
// CHECK-NEXT: }
// CHECK-NEXT: def DZ {        // X
// CHECK-NEXT:   list<int> x = [42, 42, 42];
// CHECK-NEXT: }

class X<int a, int b> {
  list<int> x = !listsplat(a, b);
}

class Y<string a, int b> {
  list<string> x = !listsplat(a, b);
}

def DX00 : X<0, 0>;
def DX01 : X<0, 1>;
def DX02 : X<0, 2>;

def DX10 : X<1, 0>;
def DX11 : X<1, 1>;
def DX12 : X<1, 2>;

def DYe0 : Y<"", 0>;
def DYe1 : Y<"", 1>;
def DYe2 : Y<"", 2>;

def DYa0 : Y<"a", 0>;
def DYa1 : Y<"a", 1>;
def DYa2 : Y<"a", 2>;

def DZ : X<42, !size([1, 2, 3])>;

#ifdef ERROR1
// ERROR1: !listsplat count -1 is negative
defvar E = !listsplat("", -1);
#endif
