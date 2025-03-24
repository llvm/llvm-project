// RUN: llvm-tblgen %s | FileCheck %s

// CHECK: class F<Y [[ARG:.+]] = ?> {
// CHECK: string ret = !if(!initialized([[ARG]].str), [[ARG]].str, "N/A");
// CHECK: }

// CHECK-LABEL: def C
// CHECK: bit c0 = 0
// CHECK: bit c1 = 1
// CHECK: bit c2 = 1
def C {
  bit c0 = !initialized(?);
  bit c1 = !initialized(0);
  bit c2 = !initialized(1);
}

class Y {
  string str = ?;
}

class F<Y y> {
  string ret = !if(!initialized(y.str), y.str, "N/A");
}

def Y0 : Y;
def Y1 : Y {
  let str = "foo";
}

// CHECK-LABEL: def FY0
// CHECK: string ret = "N/A";
// CHECK-LABEL: def FY1
// CHECK: string ret = "foo";
def FY0 : F<Y0>;
def FY1 : F<Y1>;

class G<Y y> {
  list<string> v = [y.str];
  bit isInit = !initialized(v);
}

// CHECK-LABEL: def GY0
// CHECK: isInit = 1
// CHECK-LABEL: def GY1
// CHECK: isInit = 1
def GY0 : G<Y0>;
def GY1 : G<Y1>;

class Thing;
def aThing : Thing;
class Propagate<Thing t> {
  Thing ret = !if(!initialized(t), t, ?);
}
// CHECK-LABEL: def PropagateNothing
// CHECK: Thing ret = ?
// CHECK-LABEL: def PropagateThing
// CHECK: Thing ret = aThing
def PropagateNothing : Propagate<?>;
def PropagateThing : Propagate<aThing>;
