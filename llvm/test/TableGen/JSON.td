// RUN: llvm-tblgen -dump-json %s | %python %S/JSON-check.py %s

// CHECK: data['!tablegen_json_version'] == 1

// CHECK: all(data[s]['!name'] == s for s in data if not s.startswith("!"))

class Base {}
class Intermediate : Base {}
class Derived : Intermediate {}

def D : Intermediate {}
// CHECK: 'D' in data['!instanceof']['Base']
// CHECK: 'D' in data['!instanceof']['Intermediate']
// CHECK: 'D' not in data['!instanceof']['Derived']
// CHECK: 'Base' in data['D']['!superclasses']
// CHECK: 'Intermediate' in data['D']['!superclasses']
// CHECK: 'Derived' not in data['D']['!superclasses']

def ExampleDagOp;

def FieldKeywordTest {
    int a;
    field int b;
    // CHECK: 'a' not in data['FieldKeywordTest']['!fields']
    // CHECK: 'b' in data['FieldKeywordTest']['!fields']
}

class Variables {
    int i;
    string s;
    bit b;
    bits<8> bs;
    code c;
    list<int> li;
    Base base;
    dag d;
}
def VarNull : Variables {
    // A variable not filled in at all has its value set to JSON
    // 'null', which translates to Python None
    // CHECK: data['VarNull']['i'] is None
}
def VarPrim : Variables {
    // Test initializers that map to primitive JSON types

    int i = 3;
    // CHECK: data['VarPrim']['i'] == 3

    // Integer literals should be emitted in the JSON at full 64-bit
    // precision, for the benefit of JSON readers that preserve that
    // much information. Python's is one such.
    int enormous_pos = 9123456789123456789;
    int enormous_neg = -9123456789123456789;
    // CHECK: data['VarPrim']['enormous_pos'] == 9123456789123456789
    // CHECK: data['VarPrim']['enormous_neg'] == -9123456789123456789

    string s = "hello, world";
    // CHECK: data['VarPrim']['s'] == 'hello, world'

    bit b = 0;
    // CHECK: data['VarPrim']['b'] == 0

    // bits<> arrays are stored in logical order (array[i] is the same
    // bit identified in .td files as bs{i}), which means the _visual_
    // order of the list (in default rendering) is reversed.
    bits<8> bs = { 0,0,0,1,0,1,1,1 };
    // CHECK: data['VarPrim']['bs'] == [ 1,1,1,0,1,0,0,0 ]

    code c = [{ \"  }];
    // CHECK: data['VarPrim']['c'] == r' \"  '

    list<int> li = [ 1, 2, 3, 4 ];
    // CHECK: data['VarPrim']['li'] == [ 1, 2, 3, 4 ]
}
def VarObj : Variables {
    // Test initializers that map to JSON objects containing a 'kind'
    // discriminator

    Base base = D;
    // CHECK: data['VarObj']['base']['kind'] == 'def'
    // CHECK: data['VarObj']['base']['def'] == 'D'
    // CHECK: data['VarObj']['base']['printable'] == 'D'

    dag d = (ExampleDagOp 22, "hello":$foo);
    // CHECK: data['VarObj']['d']['kind'] == 'dag'
    // CHECK: data['VarObj']['d']['operator']['kind'] == 'def'
    // CHECK: data['VarObj']['d']['operator']['def'] == 'ExampleDagOp'
    // CHECK: data['VarObj']['d']['operator']['printable'] == 'ExampleDagOp'
    // CHECK: data['VarObj']['d']['args'] == [[22, None], ["hello", "foo"]]
    // CHECK: data['VarObj']['d']['printable'] == '(ExampleDagOp 22, "hello":$foo)'

    int undef_int;
    field int ref_int = undef_int;
    // CHECK: data['VarObj']['ref_int']['kind'] == 'var'
    // CHECK: data['VarObj']['ref_int']['var'] == 'undef_int'
    // CHECK: data['VarObj']['ref_int']['printable'] == 'undef_int'

    bits<2> undef_bits;
    bits<4> ref_bits;
    let ref_bits{3-2} = 0b10;
    let ref_bits{1-0} = undef_bits{1-0};
    // CHECK: data['VarObj']['ref_bits'][3] == 1
    // CHECK: data['VarObj']['ref_bits'][2] == 0
    // CHECK: data['VarObj']['ref_bits'][1]['kind'] == 'varbit'
    // CHECK: data['VarObj']['ref_bits'][1]['var'] == 'undef_bits'
    // CHECK: data['VarObj']['ref_bits'][1]['index'] == 1
    // CHECK: data['VarObj']['ref_bits'][1]['printable'] == 'undef_bits{1}'
    // CHECK: data['VarObj']['ref_bits'][0]['kind'] == 'varbit'
    // CHECK: data['VarObj']['ref_bits'][0]['var'] == 'undef_bits'
    // CHECK: data['VarObj']['ref_bits'][0]['index'] == 0
    // CHECK: data['VarObj']['ref_bits'][0]['printable'] == 'undef_bits{0}'

    field int complex_ref_int = !add(undef_int, 2);
    // CHECK: data['VarObj']['complex_ref_int']['kind'] == 'complex'
    // CHECK: data['VarObj']['complex_ref_int']['printable'] == '!add(undef_int, 2)'
}

// Test the !anonymous member. This is tricky because when a def is
// anonymous, almost by definition, the test can't reliably predict
// the name it will be stored under! So we have to search all the defs
// in the JSON output looking for the one that has the test integer
// field set to the right value.

def Named { int AnonTestField = 1; }
// CHECK: data['Named']['AnonTestField'] == 1
// CHECK: data['Named']['!anonymous'] is False

def { int AnonTestField = 2; }
// CHECK: next(rec for rec in data.values() if isinstance(rec, dict) and rec.get('AnonTestField') == 2)['!anonymous'] is True

multiclass AnonTestMulticlass<int base> {
    def _plus_one { int AnonTestField = !add(base,1); }
    def { int AnonTestField = !add(base,2); }
}

defm NamedDefm : AnonTestMulticlass<10>;
// CHECK: data['NamedDefm_plus_one']['!anonymous'] is False
// CHECK: data['NamedDefm_plus_one']['AnonTestField'] == 11
// CHECK: next(rec for rec in data.values() if isinstance(rec, dict) and rec.get('AnonTestField') == 12)['!anonymous'] is True

// D47431 clarifies that a named def inside a multiclass gives a
// *non*-anonymous output record, even if the defm that instantiates
// that multiclass is anonymous.
defm : AnonTestMulticlass<20>;
// CHECK: next(rec for rec in data.values() if isinstance(rec, dict) and rec.get('AnonTestField') == 21)['!anonymous'] is False
// CHECK: next(rec for rec in data.values() if isinstance(rec, dict) and rec.get('AnonTestField') == 22)['!anonymous'] is True
