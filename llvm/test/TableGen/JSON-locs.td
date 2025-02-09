// RUN: llvm-tblgen -dump-json %s | %python %S/JSON-check.py %s

def Simple {}
// CHECK: data['Simple']['!locs'] == ['JSON-locs.td:3']

multiclass Multiclass1 {
  def Instance1 {}
  def Instance2 {}
}

defm DefM1 : Multiclass1;

// CHECK: data['DefM1Instance1']['!locs'] == ['JSON-locs.td:7', 'JSON-locs.td:11']
// CHECK: data['DefM1Instance2']['!locs'] == ['JSON-locs.td:8', 'JSON-locs.td:11']

multiclass Multiclass2 {
  def Instance3 {}
  def Instance4 {}
}

defm DefM2 : Multiclass1, Multiclass2;
// CHECK: data['DefM2Instance1']['!locs'] == ['JSON-locs.td:7', 'JSON-locs.td:21']
// CHECK: data['DefM2Instance2']['!locs'] == ['JSON-locs.td:8', 'JSON-locs.td:21']
// CHECK: data['DefM2Instance3']['!locs'] == ['JSON-locs.td:17', 'JSON-locs.td:21']
// CHECK: data['DefM2Instance4']['!locs'] == ['JSON-locs.td:18', 'JSON-locs.td:21']

multiclass Multiclass3 {
  defm InnerDefM : Multiclass1;
  def Instance5 {}
}

defm DefM3: Multiclass3;
// CHECK: data['DefM3InnerDefMInstance1']['!locs'] == ['JSON-locs.td:7', 'JSON-locs.td:28', 'JSON-locs.td:32']
// CHECK: data['DefM3InnerDefMInstance2']['!locs'] == ['JSON-locs.td:8', 'JSON-locs.td:28', 'JSON-locs.td:32']
// CHECK: data['DefM3Instance5']['!locs'] == ['JSON-locs.td:29', 'JSON-locs.td:32']

class BaseClass {}
class DerivedClass : BaseClass {}
// Classes do not appear in the JSON, so do not get locations.
// CHECK: 'BaseClass' not in data
// CHECK: 'DerivedClass' not in data

def ClassInstance : DerivedClass {}
// CHECK: data['ClassInstance']['!locs'] == ['JSON-locs.td:43']
