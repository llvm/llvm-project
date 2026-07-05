// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not llvm-tblgen -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not llvm-tblgen -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not llvm-tblgen -DERROR5 %s 2>&1 | FileCheck --check-prefix=ERROR5 %s

class Class<int v> {
  int value = v;
}

deftype StringAlias    = string;
deftype CodeAlias      = code;
deftype DagAlias       = dag;
deftype Boolean        = bit;
deftype Byte           = bits<8>;
deftype Integer        = int;
deftype IntList        = list<int>;
deftype ByteList       = list<Byte>;
deftype ClassList      = list<Class>;
// The type can be another type alias.
deftype ClassListAlias = ClassList;

// CHECK:      def test {
// CHECK-NEXT:   string str = "string";
// CHECK-NEXT:   string codeStr = "code";
// CHECK-NEXT:   dag dagExpr = ("string" "code");
// CHECK-NEXT:   bit bool = 0;
// CHECK-NEXT:   bits<8> byte = { 0, 1, 1, 1, 1, 0, 1, 1 };
// CHECK-NEXT:   int integer = 123;
// CHECK-NEXT:   list<int> ints = [1, 2, 3];
// CHECK-NEXT:   list<bits<8>> bytes = [{ 0, 0, 0, 0, 0, 0, 0, 1 }, { 0, 0, 0, 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 0, 0, 1, 1 }];
// CHECK-NEXT:   list<Class> defs = [anonymous_0, anonymous_1, anonymous_2];
// CHECK-NEXT: }
def test {
  StringAlias    str     = "string";
  CodeAlias      codeStr = "code";
  DagAlias       dagExpr = (str codeStr);
  Boolean        bool    = false;
  Byte           byte    = 123;
  Integer        integer = 123;
  IntList        ints    = [1, 2, 3];
  ByteList       bytes   = [1, 2, 3];
  ClassListAlias defs    = [Class<1>, Class<2>, Class<3>];
}

#ifdef ERROR1
// ERROR1: [[@LINE+1]]:9: error: type of this name 'Byte' already exists
deftype Byte = bits<8>;
#endif

#ifdef ERROR2
// ERROR2: [[@LINE+1]]:9: error: type of this name 'Class' already exists
deftype Class = int;
#endif

#ifdef ERROR3
// ERROR3: [[@LINE+1]]:22: error: cannot define type alias for class type 'Class'
deftype ClassAlias = Class;
#endif

#ifdef ERROR4
// ERROR4: [[@LINE+1]]:7: error: there is already a defined type alias 'Byte'
class Byte; // incomplete class definition.
#endif

#ifdef ERROR5
// ERROR5: [[@LINE+1]]:7: error: there is already a defined type alias 'Byte'
class Byte {}
#endif
